use std::io::{self, Write};
use std::ffi::{CString, CStr};
use std::env;
use std::path::Path;
// use std::fs::{File, OpenOptions};
// use std::os::unix::io::{AsRawFd, FromRawFd};
use nix::unistd::{fork, ForkResult, execvp, dup2};
use nix::sys::wait::waitpid;
use nix::sys::wait::WaitStatus;
use nix::fcntl::{OFlag, open};
use nix::sys::stat::Mode;
// use nix::errno::Errno;

fn main() -> io::Result<()> {
    loop {
        // Get current directory for the prompt
        let current_dir = match env::current_dir() {
            Ok(path) => {
                // Use the directory name only or ~ for home directory
                let path_str = path.to_string_lossy();
                let home = env::var("HOME").unwrap_or_default();
                
                if path_str.starts_with(&home) {
                    let path_without_home = path_str.replacen(&home, "~", 1);
                    path_without_home.to_string()
                } else {
                    // Get the directory name
                    match path.file_name() {
                        Some(name) => name.to_string_lossy().to_string(),
                        None => path_str.to_string(),
                    }
                }
            },
            Err(_) => "unknown_dir".to_string(),
        };

        // Display prompt with current directory
        print!("vssh:{} > ", current_dir);
        io::stdout().flush()?;

        // Read input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // Check for exit command
        if input == "exit" || input.is_empty() {
            break;
        }

        // Parse command and handle redirections
        let (args, redirections) = parse_command(input);
        
        if args.is_empty() {
            continue;
        }

        // Handle cd command internally
        if args[0] == "cd" {
            let new_dir = args.get(1).map_or("/", |s| s.as_str());
            
            // Handle ~ for home directory
            let target_dir = if new_dir == "~" {
                match env::var("HOME") {
                    Ok(home) => home,
                    Err(_) => continue,
                }
            } else {
                new_dir.to_string()
            };
            
            if let Err(e) = env::set_current_dir(Path::new(&target_dir)) {
                eprintln!("cd: {}: {}", target_dir, e);
            }
            continue;
        }

        // Convert to CString for execvp
        let args_cstring: Vec<CString> = args
            .iter()
            .map(|arg| CString::new(arg.clone()).unwrap())
            .collect();

        // Fork process
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                // Parent process
                match waitpid(child, None) {
                    Ok(WaitStatus::Exited(_, status)) => {
                        if status != 0 {
                            println!("Command exited with status: {}", status);
                        }
                    }
                    Ok(status) => println!("Unusual exit: {:?}", status),
                    Err(err) => println!("Error waiting for child: {}", err),
                }
            }
            Ok(ForkResult::Child) => {
                // Child process - handle redirections
                if let Err(e) = apply_redirections(&redirections) {
                    eprintln!("Redirection error: {}", e);
                    std::process::exit(1);
                }
                
                let program = CString::new(args[0].clone()).unwrap();
                let args_refs: Vec<&CStr> = args_cstring.iter().map(|arg| arg.as_c_str()).collect();
                
                match execvp(&program, &args_refs) {
                    Ok(_) => unreachable!(), // execvp replaces the current process
                    Err(err) => {
                        eprintln!("Failed to execute command: {}", err);
                        std::process::exit(1);
                    }
                }
            }
            Err(err) => {
                eprintln!("Fork failed: {}", err);
            }
        }
    }

    Ok(())
}

// Redirection types
enum Redirection {
    Input(String),           // <
    Output(String, bool),    // > (truncate) or >> (append)
    Error(String, bool),     // 2> (truncate) or 2>> (append)
    Both(String, bool),      // &> (truncate) or &>> (append)
}

// Parse command and extract redirections
fn parse_command(input: &str) -> (Vec<String>, Vec<Redirection>) {
    let mut args = Vec::new();
    let mut redirections = Vec::new();
    let mut tokens = input.split_whitespace().peekable();
    
    while let Some(token) = tokens.next() {
        match token {
            "<" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Input(file.to_string()));
                }
            },
            ">" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Output(file.to_string(), false)); // truncate
                }
            },
            ">>" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Output(file.to_string(), true)); // append
                }
            },
            "2>" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Error(file.to_string(), false)); // truncate
                }
            },
            "2>>" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Error(file.to_string(), true)); // append
                }
            },
            "&>" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Both(file.to_string(), false)); // truncate
                }
            },
            "&>>" => {
                if let Some(file) = tokens.next() {
                    redirections.push(Redirection::Both(file.to_string(), true)); // append
                }
            },
            _ => {
                // Regular argument
                args.push(token.to_string());
            }
        }
    }
    
    (args, redirections)
}

// Apply redirections in the child process
fn apply_redirections(redirections: &[Redirection]) -> Result<(), io::Error> {
    for redirection in redirections {
        match redirection {
            Redirection::Input(file) => {
                let fd = open(
                    Path::new(file),
                    OFlag::O_RDONLY,
                    Mode::empty(),
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                
                dup2(fd, 0)?; // stdin
            },
            Redirection::Output(file, append) => {
                let mut flags = OFlag::O_WRONLY | OFlag::O_CREAT;
                if *append {
                    flags |= OFlag::O_APPEND;
                } else {
                    flags |= OFlag::O_TRUNC;
                }
                
                let fd = open(
                    Path::new(file),
                    flags,
                    Mode::S_IRUSR | Mode::S_IWUSR | Mode::S_IRGRP | Mode::S_IROTH,
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                
                dup2(fd, 1)?; // stdout
            },
            Redirection::Error(file, append) => {
                let mut flags = OFlag::O_WRONLY | OFlag::O_CREAT;
                if *append {
                    flags |= OFlag::O_APPEND;
                } else {
                    flags |= OFlag::O_TRUNC;
                }
                
                let fd = open(
                    Path::new(file),
                    flags,
                    Mode::S_IRUSR | Mode::S_IWUSR | Mode::S_IRGRP | Mode::S_IROTH,
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                
                dup2(fd, 2)?; // stderr
            },
            Redirection::Both(file, append) => {
                let mut flags = OFlag::O_WRONLY | OFlag::O_CREAT;
                if *append {
                    flags |= OFlag::O_APPEND;
                } else {
                    flags |= OFlag::O_TRUNC;
                }
                
                let fd = open(
                    Path::new(file),
                    flags,
                    Mode::S_IRUSR | Mode::S_IWUSR | Mode::S_IRGRP | Mode::S_IROTH,
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                
                dup2(fd, 1)?; // stdout
                dup2(fd, 2)?; // stderr
            },
        }
    }
    
    Ok(())
}