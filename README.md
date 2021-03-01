Stats CLI
=========

[![crates.io](https://img.shields.io/crates/v/stats-cli)](https://crates.io/crates/stats-cli)
[![documentation](https://docs.rs/stats-cli/badge.svg)](https://docs.rs/stats-cli)
[![tests](https://github.com/erikbrinkman/stats/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/stats/actions/workflows/rust.yml)
![MIT License](https://img.shields.io/github/license/erikbrinkman/stats)

Rust package for efficiently computing statistics on floating point numbers from the command line.
This contains both a command line binary and a rust library with incremental statistics objects.
Currently only summary statistics (min, mean, variance), percentiles, and mode are supported.


To Do
-----

- [ ] Add support for generic numeric data types, after they're more supported in rust.
- [ ] Add sketches.
- [ ] Add testing to cli
