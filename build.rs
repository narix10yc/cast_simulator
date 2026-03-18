use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const CPU_FFI_SOURCES: &[&str] = &[
    "src/cpp/cpu.h",
    "src/cpp/cpu.cpp",
    "src/cpp/cpu_gen.h",
    "src/cpp/cpu_gen.cpp",
    "src/cpp/cpu_jit.h",
    "src/cpp/cpu_jit.cpp",
    "src/cpp/cpu_util.h",
];

const CUDA_FFI_SOURCES: &[&str] = &[
    "src/cpp/cuda.h",
    "src/cpp/cuda.cpp",
    "src/cpp/cuda_gen.h",
    "src/cpp/cuda_gen.cpp",
    "src/cpp/cuda_jit.h",
    "src/cpp/cuda_jit.cpp",
    "src/cpp/cuda_util.h",
    "src/cpp/cuda_exec.cpp",
];

const CPU_LLVM_COMPONENTS: &[&str] = &["core", "orcjit", "native", "passes"];
const CUDA_LLVM_COMPONENTS: &[&str] = &["core", "nvptx", "passes"];

fn main() {
    emit_rerun_if_changed(CPU_FFI_SOURCES);
    emit_rerun_if_env_changed(&["CXX", "LLVM_CONFIG", "HOMEBREW_PREFIX"]);

    let out_dir = required_env_path("OUT_DIR");
    let llvm_config = llvm_config_path();

    build_cpu_ffi(&out_dir, &llvm_config);

    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        emit_rerun_if_changed(CUDA_FFI_SOURCES);
        emit_rerun_if_env_changed(&["CUDA_PATH"]);
        build_cuda_ffi(&out_dir, &llvm_config);
    }
}

fn build_cpu_ffi(out_dir: &Path, llvm_config: &Path) {
    let llvm = llvm_build_config(llvm_config, CPU_LLVM_COMPONENTS);
    let archive = compile_cpp_archive(
        out_dir,
        &[
            "src/cpp/cpu.cpp",
            "src/cpp/cpu_gen.cpp",
            "src/cpp/cpu_jit.cpp",
        ],
        "libcast_cpu_ffi.a",
        "",
        &llvm.cxxflags,
        &[],
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={archive}");
    emit_llvm_link_flags(&llvm.link_flags);
    println!("cargo:rustc-link-lib={}", cxx_stdlib_name());
}

fn build_cuda_ffi(out_dir: &Path, llvm_config: &Path) {
    let llvm = llvm_build_config(llvm_config, CUDA_LLVM_COMPONENTS);
    let mut extra_includes = Vec::new();
    if let Some(cuda_include) = cuda_include_dir() {
        extra_includes.push(cuda_include);
    }

    let archive = compile_cpp_archive(
        out_dir,
        &[
            "src/cpp/cuda.cpp",
            "src/cpp/cuda_gen.cpp",
            "src/cpp/cuda_jit.cpp",
            "src/cpp/cuda_exec.cpp",
        ],
        "libcast_cuda_ffi.a",
        "_cuda",
        &llvm.cxxflags,
        &extra_includes,
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={archive}");
    emit_llvm_link_flags(&llvm.link_flags);

    // The CUDA driver does not come from llvm-config; find it via toolkit path.
    if let Some(cuda_lib) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        let stubs_dir = cuda_lib.join("stubs");
        if stubs_dir.exists() {
            println!("cargo:rustc-link-search=native={}", stubs_dir.display());
        }
    }

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib={}", cxx_stdlib_name());
}

fn compile_cpp_archive(
    out_dir: &Path,
    sources: &[&str],
    archive_filename: &str,
    object_suffix: &str,
    llvm_cxxflags: &[String],
    extra_includes: &[PathBuf],
) -> String {
    let mut object_files = Vec::new();

    for source in sources {
        let object_file = object_path(out_dir, source, object_suffix);
        let mut command = cxx_compiler();
        command.arg("-std=c++17").arg("-fPIC").arg("-fexceptions");
        append_filtered_llvm_cxxflags(&mut command, llvm_cxxflags);
        for include_dir in extra_includes {
            command.arg(format!("-I{}", include_dir.display()));
        }
        command.arg("-c").arg(source).arg("-o").arg(&object_file);
        run(&mut command);
        object_files.push(object_file);
    }

    let archive_path = out_dir.join(archive_filename);
    let mut ar = Command::new("ar");
    ar.arg("crus").arg(&archive_path);
    for object_file in &object_files {
        ar.arg(object_file);
    }
    run(&mut ar);

    archive_path
        .file_stem()
        .expect("archive should have a file stem")
        .to_string_lossy()
        .trim_start_matches("lib")
        .to_owned()
}

fn append_filtered_llvm_cxxflags(command: &mut Command, llvm_cxxflags: &[String]) {
    for flag in llvm_cxxflags {
        // We intentionally keep exceptions enabled in the bridge code and set
        // the language mode ourselves to avoid LLVM overriding it.
        if flag == "-std=c++17" || flag == "-fno-exceptions" {
            continue;
        }
        command.arg(flag);
    }
}

fn object_path(out_dir: &Path, source: &str, suffix: &str) -> PathBuf {
    let stem = Path::new(source)
        .file_stem()
        .unwrap_or_else(|| panic!("source file has no stem: {source}"))
        .to_string_lossy();
    out_dir.join(format!("{stem}{suffix}.o"))
}

struct LlvmBuildConfig {
    cxxflags: Vec<String>,
    link_flags: Vec<String>,
}

fn llvm_build_config(llvm_config: &Path, components: &[&str]) -> LlvmBuildConfig {
    let mut link_args = vec!["--system-libs", "--libs"];
    link_args.extend_from_slice(components);

    let cxxflags = llvm_config_flags(llvm_config, &["--cxxflags"]);
    let mut link_flags = llvm_config_flags(llvm_config, &["--ldflags"]);
    link_flags.extend(llvm_config_flags(llvm_config, &link_args));

    LlvmBuildConfig {
        cxxflags,
        link_flags,
    }
}

/// Emits Cargo link directives from llvm-config output.
///
/// LLVM reports both search paths and bare `-l...` flags. On macOS/Homebrew,
/// some "system" libraries such as zstd may live under the Homebrew prefix
/// while llvm-config only reports `-lzstd`. In that case we add the Homebrew
/// library directory explicitly so the final Rust link step succeeds without a
/// shell-level `LIBRARY_PATH` workaround.
fn emit_llvm_link_flags(flags: &[String]) {
    let mut search_paths = Vec::<PathBuf>::new();
    let mut linked_libs = Vec::<String>::new();

    for flag in flags {
        if let Some(path) = flag.strip_prefix("-L") {
            let path = PathBuf::from(path);
            println!("cargo:rustc-link-search=native={}", path.display());
            search_paths.push(path);
            continue;
        }
        if let Some(lib) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
            linked_libs.push(lib.to_owned());
        }
    }

    emit_platform_link_search_fallbacks(&linked_libs, &search_paths);
}

fn emit_platform_link_search_fallbacks(linked_libs: &[String], search_paths: &[PathBuf]) {
    if !cfg!(target_os = "macos") || !linked_libs.iter().any(|lib| lib == "zstd") {
        return;
    }

    for candidate in macos_library_search_candidates("zstd") {
        if search_paths.iter().any(|existing| existing == &candidate) {
            return;
        }
        println!("cargo:rustc-link-search=native={}", candidate.display());
        return;
    }
}

fn macos_library_search_candidates(lib_stem: &str) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(prefix) = env::var_os("HOMEBREW_PREFIX") {
        candidates.push(PathBuf::from(prefix).join("lib"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/lib"));
    candidates.push(PathBuf::from("/usr/local/lib"));

    candidates
        .into_iter()
        .filter(|path| {
            path.join(format!("lib{lib_stem}.dylib")).exists()
                || path.join(format!("lib{lib_stem}.a")).exists()
        })
        .collect()
}

fn emit_rerun_if_changed(paths: &[&str]) {
    for path in paths {
        println!("cargo:rerun-if-changed={path}");
    }
}

fn emit_rerun_if_env_changed(vars: &[&str]) {
    for var in vars {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// Returns the configured C++ compiler, defaulting to `c++`.
fn cxx_compiler() -> Command {
    if let Some(cxx) = env::var_os("CXX") {
        return Command::new(cxx);
    }
    Command::new("c++")
}

/// Returns the C++ standard library name for the current Rust target.
fn cxx_stdlib_name() -> &'static str {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        "c++"
    } else {
        "stdc++"
    }
}

/// Locates llvm-config from the required LLVM_CONFIG environment variable.
fn llvm_config_path() -> PathBuf {
    let path = required_env_path("LLVM_CONFIG");
    if !path.is_file() {
        panic!(
            "LLVM_CONFIG points to `{}`, but that path is not a file",
            path.display()
        );
    }
    path
}

/// Runs llvm-config with `args` and returns the whitespace-split stdout.
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
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "`{}` with args {:?} failed with {}{}{}",
            llvm_config.display(),
            args,
            output.status,
            if stderr.trim().is_empty() { "" } else { ": " },
            stderr.trim()
        );
    }

    String::from_utf8(output.stdout)
        .unwrap_or_else(|_| panic!("`{}` produced non-UTF-8 output", llvm_config.display()))
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect()
}

/// Returns the CUDA toolkit include directory.
fn cuda_include_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_PATH") {
        return Some(PathBuf::from(path).join("include"));
    }

    for candidate in ["/usr/local/cuda/include", "/usr/cuda/include"] {
        let candidate = PathBuf::from(candidate);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Returns the CUDA toolkit lib64 directory.
fn cuda_lib_dir() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_PATH") {
        return Some(PathBuf::from(path).join("lib64"));
    }

    for candidate in ["/usr/local/cuda/lib64", "/usr/cuda/lib64"] {
        let candidate = PathBuf::from(candidate);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn required_env_path(var: &str) -> PathBuf {
    env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        panic!(
            "{var} must be set{}",
            if var == "LLVM_CONFIG" {
                " (for example: export LLVM_CONFIG=$HOME/llvm/22.1.1/release-install/bin/llvm-config)"
            } else {
                ""
            }
        )
    })
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
