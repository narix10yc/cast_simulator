use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Rebuild when C++ sources change.
    for path in [
        "src/cpp/cpu.h",
        "src/cpp/cpu.cpp",
        "src/cpp/cpu_gen.h",
        "src/cpp/cpu_gen.cpp",
        "src/cpp/cpu_jit.h",
        "src/cpp/cpu_jit.cpp",
        "src/cpp/cpu_util.h",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    // Rebuild when the compiler or LLVM installation changes.
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    let llvm_config = llvm_config_path();

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_ffi(&out_dir, &llvm_config);
    }
    let llvm_cxxflags = llvm_config_flags(&llvm_config, &["--cxxflags"]);
    let llvm_ldflags = llvm_config_flags(&llvm_config, &["--ldflags"]);
    let llvm_libs = llvm_config_flags(
        &llvm_config,
        &[
            "--system-libs",
            "--libs",
            "core",
            "orcjit",
            "native",
            "passes",
        ],
    );

    let mut objects = Vec::new();
    for source in [
        "src/cpp/cpu.cpp",
        "src/cpp/cpu_gen.cpp",
        "src/cpp/cpu_jit.cpp",
    ] {
        let object = out_dir.join(
            Path::new(source)
                .file_name()
                .expect("source file should have a file name")
                .to_string_lossy()
                .replace(".cpp", ".o"),
        );
        let mut command = cxx_compiler();
        command.arg("-std=c++17").arg("-fPIC").arg("-fexceptions");
        for flag in &llvm_cxxflags {
            if flag == "-std=c++17" || flag == "-fno-exceptions" {
                continue;
            }
            command.arg(flag);
        }
        command.arg("-c").arg(source).arg("-o").arg(&object);
        run(&mut command);
        objects.push(object);
    }

    let lib = out_dir.join("libcast_cpu_ffi.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for object in &objects {
        ar.arg(object);
    }
    run(&mut ar);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cast_cpu_ffi");
    for flag in llvm_ldflags.iter().chain(llvm_libs.iter()) {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    println!("cargo:rustc-link-lib={}", cxx_stdlib_name());
}

/// Returns a `Command` for the C++ compiler, respecting the `CXX` environment variable.
fn cxx_compiler() -> Command {
    if let Some(cxx) = env::var_os("CXX") {
        return Command::new(cxx);
    }
    Command::new("c++")
}

/// Returns the system C++ standard library name for the current target.
fn cxx_stdlib_name() -> &'static str {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        "c++"
    } else {
        "stdc++"
    }
}

/// Locates `llvm-config` from the required `LLVM_CONFIG` environment variable.
fn llvm_config_path() -> PathBuf {
    env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("LLVM_CONFIG must be set to the path of the llvm-config binary"))
}

/// Runs `llvm-config` with `args` and returns the whitespace-split output tokens.
fn llvm_config_flags(llvm_config: &Path, args: &[&str]) -> Vec<String> {
    let output = Command::new(llvm_config)
        .args(args)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run `{}` with args {:?}: {}",
                llvm_config.display(),
                args,
                err
            )
        });
    if !output.status.success() {
        panic!(
            "`{}` with args {:?} failed with {}",
            llvm_config.display(),
            args,
            output.status
        );
    }
    String::from_utf8(output.stdout)
        .expect("llvm-config output should be utf-8")
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

/// Compiles the CUDA FFI layer (cuda.cpp, cuda_gen.cpp, cuda_jit.cpp) into
/// libcast_cuda_ffi.a and emits the necessary link directives.
fn build_cuda_ffi(out_dir: &Path, llvm_config: &Path) {
    for path in [
        "src/cpp/cuda.h",
        "src/cpp/cuda.cpp",
        "src/cpp/cuda_gen.h",
        "src/cpp/cuda_gen.cpp",
        "src/cpp/cuda_jit.h",
        "src/cpp/cuda_jit.cpp",
        "src/cpp/cuda_util.h",
        "src/cpp/cuda_exec.cpp",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVJITLINK_LIB");

    // NVPTX needs: core + nvptx* + passes; NOT orcjit or native.
    let llvm_cxxflags = llvm_config_flags(llvm_config, &["--cxxflags"]);
    let llvm_ldflags = llvm_config_flags(llvm_config, &["--ldflags"]);
    let llvm_libs = llvm_config_flags(
        llvm_config,
        &["--system-libs", "--libs", "core", "nvptx", "passes"],
    );

    let mut objects = Vec::new();
    for source in [
        "src/cpp/cuda.cpp",
        "src/cpp/cuda_gen.cpp",
        "src/cpp/cuda_jit.cpp",
        "src/cpp/cuda_exec.cpp",
    ] {
        let object = out_dir.join(
            Path::new(source)
                .file_name()
                .expect("source file should have a file name")
                .to_string_lossy()
                .replace(".cpp", "_cuda.o"),
        );
        let mut command = cxx_compiler();
        command.arg("-std=c++17").arg("-fPIC").arg("-fexceptions");
        for flag in &llvm_cxxflags {
            if flag == "-std=c++17" || flag == "-fno-exceptions" {
                continue;
            }
            command.arg(flag);
        }
        if let Some(cuda_inc) = cuda_include_dir() {
            command.arg(format!("-I{}", cuda_inc.display()));
        }
        command.arg("-c").arg(source).arg("-o").arg(&object);
        run(&mut command);
        objects.push(object);
    }

    let lib = out_dir.join("libcast_cuda_ffi.a");
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&lib);
    for object in &objects {
        ar.arg(object);
    }
    run(&mut ar);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cast_cuda_ffi");
    for flag in llvm_ldflags.iter().chain(llvm_libs.iter()) {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    // nvJitLink + CUDA driver: check NVJITLINK_LIB env first, then common CUDA
    // toolkit paths.  libcuda (the driver) is usually in the standard library
    // search path on systems with an NVIDIA driver; the stubs directory under
    // lib64 provides a link-time stub on driver-less build machines.
    if let Ok(nvjl) = env::var("NVJITLINK_LIB") {
        println!("cargo:rustc-link-search=native={nvjl}");
    } else if let Some(d) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", d.display());
        let stubs = d.join("stubs");
        if stubs.exists() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
    }
    println!("cargo:rustc-link-lib=nvJitLink");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib={}", cxx_stdlib_name());
}

/// Returns the CUDA toolkit include directory, checking `CUDA_PATH` first then
/// common installation paths.
fn cuda_include_dir() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        return Some(PathBuf::from(p).join("include"));
    }
    for candidate in ["/usr/local/cuda/include", "/usr/cuda/include"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Returns the CUDA toolkit lib64 directory.
fn cuda_lib_dir() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        return Some(PathBuf::from(p).join("lib64"));
    }
    for candidate in ["/usr/local/cuda/lib64", "/usr/cuda/lib64"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Runs `command`, panicking with a descriptive message on failure.
fn run(command: &mut Command) {
    let program = command.get_program().to_string_lossy().into_owned();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run `{}` with args {:?}: {}", program, args, err));
    if !status.success() {
        panic!(
            "command `{}` with args {:?} failed with {}",
            program, args, status
        );
    }
}
