use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};

struct Shell {
    current_dir: PathBuf,
    previous_dir: Option<PathBuf>,
    env_vars: HashMap<String, String>,
    running: bool,
    background_pids: Vec<u32>,
}

impl Shell {
    fn new() -> Self {
        Shell {
            current_dir: env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            previous_dir: None,
            env_vars: HashMap::new(),
            running: true,
            background_pids: Vec::new(),
        }
    }

    fn run(&mut self) {
        ctrlc::set_handler(move || {
            println!("\nType 'exit' to quit.");
        }).expect("Error setting Ctrl-C handler");

        while self.running {
            self.check_background_processes();
            print!("{}> ", self.current_dir.display());
            io::stdout().flush().unwrap();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                eprintln!("Failed to read line");
                continue;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            self.execute_command(input);
        }
    }

    fn check_background_processes(&mut self) {
        let mut finished = Vec::new();
        for &pid in &self.background_pids {
            if let Ok(WaitStatus::Exited(_, _)) = waitpid(nix::unistd::Pid::from_raw(pid as i32), Some(WaitPidFlag::WNOHANG)) {
                finished.push(pid);
            }
        }
        self.background_pids.retain(|pid| !finished.contains(pid));
    }

    fn execute_command(&mut self, command: &str) {
        // Background execution
        let background = command.ends_with(" &");
        let command = if background {
            command[..command.len() - 2].trim()
        } else {
            command
        };

        // Pipe support
        let pipe_segments: Vec<&str> = command.split('|').map(|s| s.trim()).collect();
        if pipe_segments.len() == 1 {
            self.process_command(pipe_segments[0], background);
        } else {
            self.process_piped_commands(&pipe_segments, background);
        }
    }

    fn process_command(&mut self, command: &str, background: bool) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        // Handle simple env assignment like VAR=value
        if parts[0].contains('=') && !parts[0].starts_with('=') {
            let mut split = parts[0].splitn(2, '=');
            let key = split.next().unwrap().to_string();
            let value = split.next().unwrap().to_string();
            self.env_vars.insert(key, value);
            return;
        }

        match parts[0] {
            "cd" => self.change_directory(parts.get(1).map(|s| *s)),
            "exit" => self.running = false,
            "pwd" => println!("{}", self.current_dir.display()),
            _ => {
                let (cmd, stdin, stdout) = self.parse_redirections(command);
                self.execute_external_command(&cmd, stdin, stdout, background);
            }
        }
    }

    fn change_directory(&mut self, dir: Option<&str>) {
        let new_dir = match dir {
            None | Some("~") => dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")),
            Some("-") => {
                if let Some(prev) = &self.previous_dir {
                    prev.clone()
                } else {
                    eprintln!("No previous directory");
                    return;
                }
            }
            Some(path) => {
                let p = Path::new(path);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    self.current_dir.join(p)
                }
            }
        };

        if let Ok(canonical) = std::fs::canonicalize(&new_dir) {
            if canonical.is_dir() {
                self.previous_dir = Some(self.current_dir.clone());
                self.current_dir = canonical;
                env::set_current_dir(&self.current_dir).unwrap_or_else(|e| {
                    eprintln!("Failed to change directory: {}", e);
                });
            } else {
                eprintln!("Not a directory: {}", new_dir.display());
            }
        } else {
            eprintln!("Invalid path: {}", new_dir.display());
        }
    }

    fn parse_redirections(&self, command: &str) -> (String, Option<String>, Option<String>) {
        let mut parts = shell_words::split(command).unwrap_or_default();
        let mut stdin = None;
        let mut stdout = None;
        let mut cmd_parts = Vec::new();

        let mut i = 0;
        while i < parts.len() {
            match parts[i].as_str() {
                ">" => {
                    if i + 1 < parts.len() {
                        stdout = Some(parts[i + 1].clone());
                        i += 1;
                    }
                }
                "<" => {
                    if i + 1 < parts.len() {
                        stdin = Some(parts[i + 1].clone());
                        i += 1;
                    }
                }
                _ => cmd_parts.push(parts[i].clone()),
            }
            i += 1;
        }

        (cmd_parts.join(" "), stdin, stdout)
    }

    fn expand_variables(&self, input: &str) -> String {
        let mut result = String::new();
        let mut chars = input.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '$' {
                let mut var = String::new();
                while let Some(&next) = chars.peek() {
                    if next.is_alphanumeric() || next == '_' {
                        var.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Some(value) = self.env_vars.get(&var) {
                    result.push_str(value);
                } else if let Ok(env_value) = env::var(&var) {
                    result.push_str(&env_value);
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    fn process_piped_commands(&mut self, commands: &[&str], background: bool) {
        let mut previous_stdout = None;
        let mut processes = Vec::new();

        for (i, cmd) in commands.iter().enumerate() {
            let (parsed_cmd, stdin_file, stdout_file) = self.parse_redirections(cmd);
            let expanded_cmd = self.expand_variables(&parsed_cmd);
            let parts: Vec<&str> = expanded_cmd.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            let mut command = Command::new(parts[0]);
            command.args(&parts[1..]);
            command.current_dir(&self.current_dir);

            if i == 0 && stdin_file.is_some() {
                let file = File::open(stdin_file.unwrap()).unwrap_or_else(|e| {
                    eprintln!("Failed to open input file: {}", e);
                    std::process::exit(1);
                });
                command.stdin(Stdio::from(file));
            } else if let Some(prev_stdout) = previous_stdout.take() {
                command.stdin(prev_stdout);
            }

            if i == commands.len() - 1 && stdout_file.is_some() {
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(stdout_file.unwrap())
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to open output file: {}", e);
                        std::process::exit(1);
                    });
                command.stdout(Stdio::from(file));
            } else if i < commands.len() - 1 {
                command.stdout(Stdio::piped());
            }

            let mut child = command.spawn().unwrap_or_else(|e| {
                eprintln!("Failed to execute command: {}", e);
                std::process::exit(1);
            });

            if i < commands.len() - 1 {
                previous_stdout = child.stdout.take().map(Stdio::from);
            }

            processes.push(child);
        }

        if !background {
            for mut child in processes {
                let _ = child.wait();
            }
        } else if let Some(child) = processes.last() {
            println!("[{}] Running in background", child.id());
            self.background_pids.push(child.id());
        }
    }

    fn execute_external_command(
        &mut self,
        command: &str,
        stdin_file: Option<String>,
        stdout_file: Option<String>,
        background: bool,
    ) {
        let expanded = self.expand_variables(command);
        let parts: Vec<&str> = expanded.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        let mut cmd = Command::new(parts[0]);
        cmd.args(&parts[1..]);
        cmd.current_dir(&self.current_dir);

        if let Some(input_file) = stdin_file {
            if let Ok(file) = File::open(&input_file) {
                cmd.stdin(Stdio::from(file));
            } else {
                eprintln!("Failed to open input file: {}", input_file);
                return;
            }
        }

        if let Some(output_file) = stdout_file {
            if let Ok(file) = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&output_file)
            {
                cmd.stdout(Stdio::from(file));
            } else {
                eprintln!("Failed to open output file: {}", output_file);
                return;
            }
        }

        match cmd.spawn() {
            Ok(mut child) => {
                if background {
                    println!("[{}] Running in background", child.id());
                    self.background_pids.push(child.id());
                } else {
                    if let Err(e) = child.wait() {
                        eprintln!("Failed to wait for command: {}", e);
                    }
                }
            }
            Err(e) => eprintln!("Failed to execute command: {}", e),
        }
    }
}

fn main() {
    println!("Simple Rust Shell - Type 'exit' to quit");
    let mut shell = Shell::new();
    shell.run();
}
