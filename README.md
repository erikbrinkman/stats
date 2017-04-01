Stats CLI
=========

Rust package for efficiently computing statistics on floating point numbers from the command line.
This contains both a command line binary and a rust library with incremental statistics objects.
Currently only summary statistics (min, mean, variance), percentiles, and mode are supported.


To Do
-----

- Add support for generic numeric data types, after they're more supported in rust.
- Add sketches.
