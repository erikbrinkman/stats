Stats CLI
=========

[![Crates.io](https://img.shields.io/crates/v/stats-cli.svg)](https://crates.io/crates/stats-cli)
[![Documentation](https://docs.rs/stats-cli/badge.svg)](https://docs.rs/stats-cli)
[![Build Status](https://github.com/erikbrinkman/stats/workflows/build/badge.svg)](https://github.com/erikbrinkman/stats/actions)

Rust package for efficiently computing statistics on floating point numbers from the command line.
This contains both a command line binary and a rust library with incremental statistics objects.
Currently only summary statistics (min, mean, variance), percentiles, and mode are supported.


To Do
-----

- [ ] Add support for generic numeric data types, after they're more supported in rust.
- [ ] Add sketches.
- [ ] Add testing to cli
