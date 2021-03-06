use std::f64;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, StdinLock, StdoutLock, Write};
use std::iter;

use clap::{crate_version, App, Arg, ArgGroup, ArgMatches};

use inc_stats::{Mode, Percentiles, SummStats};

fn parse_command_line<'a>() -> ArgMatches<'a> {
    App::new("stats")
        .version(crate_version!())
        .author("Erik Brinkman <erik.brinkman@gmail.com>")
        .about("Compute summary statistics of streams of numbers")
        // Output
        .arg(Arg::with_name("count")
             .short("c")
             .long("count")
             .help("Print count"))
        .arg(Arg::with_name("min")
             .long("min")
             .help("Print min"))
        .arg(Arg::with_name("max")
             .long("max")
             .help("Print max"))
        .arg(Arg::with_name("mean")
             .short("m")
             .long("mean")
             .help("Print mean"))
        .arg(Arg::with_name("sum")
             .long("sum")
             .help("Print sum"))
        .arg(Arg::with_name("stddev")
             .short("s")
             .long("stddev")
             .help("Print sample standard deviation"))
        .arg(Arg::with_name("var")
             .short("v")
             .long("var")
             .help("Print sample variance"))
        .arg(Arg::with_name("stderr")
             .long("stderr")
             .help("Print standard error"))
        .arg(Arg::with_name("median")
             .long("median")
             .help("Print median (Note: computing median takes O(n) space"))
        .arg(Arg::with_name("percentiles")
             .short("p")
             .long("percentiles")
             .takes_value(true)
             .use_delimiter(true)
             .help("Print arbitrary percentiles. The argument should be a comma delimited list of floats in [0, 100] (Note: computing any percentile takes O(n) space)"))
        .arg(Arg::with_name("mode")
             .long("mode")
             .help("Print mode (Note: computing mode takes O(n) space"))
        .arg(Arg::with_name("mode-count")
             .long("mode-count")
             .help("Print the number of times the mode occured (Note: computing mode takes O(n) space"))
        .arg(Arg::with_name("distinct")
             .long("distinct")
             .help("Print the number of distinct values (Note: computing mode takes O(n) space"))
        .arg(Arg::with_name("distinct-nan")
             .long("distinct-nan")
             .help("Print the number of distinct values treading all NaNs as distinct (Note: computing mode takes O(n) space"))
        // Formatting
        .group(ArgGroup::with_name("format").args(&["tsv", "json"]))
        .arg(Arg::with_name("tsv")
             .short("t")
             .long("tsv")
             .help("Output as tsv. This will force tsv output even if only one statistics is requested."))
        .arg(Arg::with_name("json")
             .short("j")
             .long("json")
             .help("Output as compressed json"))
        // IO
        .arg(Arg::with_name("input")
             .short("i")
             .long("input")
             .value_name("file")
             .default_value("-")
             .help("Take input from file"))
        .arg(Arg::with_name("output")
             .short("o")
             .long("output")
             .value_name("file")
             .default_value("-")
             .help("Write output to file"))
        .get_matches()
}

// TODO this technique is probablly better handled by enum_dispatch but it doesn't work for these
// types at the current moment
enum StatsReader<'a> {
    Stdin(StdinLock<'a>),
    File(BufReader<File>),
}

impl<'a> Read for StatsReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            StatsReader::Stdin(lock) => lock.read(buf),
            StatsReader::File(file) => file.read(buf),
        }
    }
}

impl<'a> BufRead for StatsReader<'a> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            StatsReader::Stdin(lock) => lock.fill_buf(),
            StatsReader::File(file) => file.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            StatsReader::Stdin(lock) => lock.consume(amt),
            StatsReader::File(file) => file.consume(amt),
        }
    }
}

enum StatsWriter<'a> {
    Stdout(StdoutLock<'a>),
    File(BufWriter<File>),
}

impl<'a> Write for StatsWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            StatsWriter::Stdout(lock) => lock.write(buf),
            StatsWriter::File(file) => file.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            StatsWriter::Stdout(lock) => lock.flush(),
            StatsWriter::File(file) => file.flush(),
        }
    }
}

fn main() {
    let matches = parse_command_line();

    // Setup input
    let stdin = io::stdin();
    let input = match matches.value_of("input") {
        Some("-") => StatsReader::Stdin(stdin.lock()),
        Some(file_name) => StatsReader::File(BufReader::new(
            File::open(file_name)
                .unwrap_or_else(|_| panic!("File \"{}\" does not exist", file_name)),
        )),
        None => unreachable!(),
    };

    // Setup output
    let stdout = io::stdout();
    let mut output = match matches.value_of("output") {
        Some("-") => StatsWriter::Stdout(stdout.lock()),
        Some(file_name) => {
            StatsWriter::File(BufWriter::new(File::open(file_name).unwrap_or_else(|_| {
                panic!("Couldn't open file \"{}\" for writing", file_name)
            })))
        }
        None => unreachable!(),
    };

    // Process input
    let mut stats = SummStats::new();
    let mut percs = Percentiles::new();
    let mut mode = Mode::new();

    let add_mode = ["mode", "mode-count", "distinct", "distinct-nan"]
        .iter()
        .any(|s| matches.is_present(s));
    let add_percs = ["percentiles", "median"]
        .iter()
        .any(|s| matches.is_present(s));
    let add_stats = [
        "count", "min", "max", "mean", "sum", "stddev", "var", "stderr",
    ]
    .iter()
    .any(|s| matches.is_present(s))
        || !(add_mode && add_percs);

    for line in input.lines() {
        for token in line
            .expect("Couldn't read from file")
            .split(char::is_whitespace)
            .filter(|s| !s.is_empty())
        {
            let num: f64 = token
                .parse()
                .unwrap_or_else(|_| panic!("Could not parse \"{}\" as float", token));
            if add_mode {
                mode.add(num);
            }
            if add_percs {
                percs.add(num);
            }
            if add_stats {
                stats.add(num);
            }
        }
    }

    // Generate output
    let results = compute_results(&matches, &stats, &mut percs, &mode);

    // Write output
    if matches.is_present("tsv") {
        write_tsv(&results, &mut output);
    } else if matches.is_present("json") {
        write_json(&results, &mut output);
    } else if results.len() == 1 {
        let (_, val) = results[0];
        writeln!(output, "{}", val).expect("couldn't write to output");
    } else {
        write_tsv(&results, &mut output);
    }
}

fn compute_results(
    matches: &ArgMatches,
    stats: &SummStats<f64>,
    percs: &mut Percentiles<f64>,
    mode: &Mode<f64>,
) -> Vec<(String, f64)> {
    let mut results = Vec::new();
    if matches.is_present("count") {
        results.push((String::from("count"), stats.count() as f64));
    }
    if matches.is_present("min") {
        results.push((String::from("min"), stats.min().unwrap_or(f64::NAN)));
    }
    if matches.is_present("max") {
        results.push((String::from("max"), stats.max().unwrap_or(f64::NAN)));
    }
    if matches.is_present("mean") {
        results.push((String::from("mean"), stats.mean().unwrap_or(f64::NAN)));
    }
    if matches.is_present("sum") {
        results.push((String::from("sum"), stats.sum()));
    }
    if matches.is_present("stddev") {
        results.push((
            String::from("stddev"),
            stats.standard_deviation().unwrap_or(f64::NAN),
        ));
    }
    if matches.is_present("var") {
        results.push((String::from("var"), stats.variance().unwrap_or(f64::NAN)));
    }
    if matches.is_present("stderr") {
        results.push((
            String::from("stderr"),
            stats.standard_error().unwrap_or(f64::NAN),
        ));
    }
    if matches.is_present("percentiles") {
        let percentiles: Vec<f64> = matches
            .values_of("percentiles")
            .unwrap()
            .map(|p| {
                p.parse::<f64>()
                    .unwrap_or_else(|_| panic!("Could not parse \"{}\" as float", p))
            })
            .collect();
        let vals = match percs.percentiles(percentiles.iter().map(|p| p / 100.0)) {
            Err(_) => panic!("percentiles must between 0 and 100: {:?}", percentiles),
            Ok(None) => iter::repeat(f64::NAN).take(percentiles.len()).collect(),
            Ok(Some(pvals)) => pvals,
        };
        for (perc, val) in percentiles.iter().zip(vals) {
            results.push((format!("{}%", perc), val));
        }
    }
    if matches.is_present("median") {
        results.push((String::from("median"), percs.median().unwrap_or(f64::NAN)));
    }
    if matches.is_present("mode") {
        results.push((String::from("mode"), mode.mode().unwrap_or(f64::NAN)));
    }
    if matches.is_present("mode-count") {
        results.push((String::from("mode #"), mode.mode_count() as f64));
    }
    if matches.is_present("distinct") {
        results.push((String::from("distinct"), mode.count_distinct() as f64));
    }
    if matches.is_present("distinct-nan") {
        results.push((
            String::from("distinct (NaN)"),
            mode.count_distinct_nan() as f64,
        ));
    }

    // Defaults
    if results.is_empty() {
        results.push((String::from("count"), stats.count() as f64));
        results.push((String::from("min"), stats.min().unwrap_or(f64::NAN)));
        results.push((String::from("max"), stats.max().unwrap_or(f64::NAN)));
        results.push((String::from("mean"), stats.mean().unwrap_or(f64::NAN)));
        results.push((
            String::from("stddev"),
            stats.standard_deviation().unwrap_or(f64::NAN),
        ));
    }

    results
}

fn write_tsv(results: &[(String, f64)], output: &mut impl Write) {
    for &(ref name, ref val) in results {
        writeln!(output, "{}\t{}", name, val).expect("couldn't write to output");
    }
}

fn write_json(results: &[(String, f64)], output: &mut impl Write) {
    write!(output, "{{").expect("couldn't write to output");
    let mut iter = results.iter();
    let &(ref name, ref val) = iter.next().unwrap();
    write!(output, "\"{}\":{}", name, val).expect("couldn't write to output");
    for &(ref name, ref val) in iter {
        write!(output, ",\"{}\":{}", name, val).expect("couldn't write to output");
    }
    writeln!(output, "}}").expect("couldn't write to output");
}
