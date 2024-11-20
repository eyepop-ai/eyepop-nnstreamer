---
title: EyePop
...

# Installing NNStreamer for EyePop development

These instructions supplement the docs for installing the nnstreamer package for Ubuntu:
* [With an existing .deb package](./getting-started-ubuntu-ppa.md)
* [By building a .deb package locally](./getting-started-ubuntu-debuild.md)
* [By building through meson/ninja](./getting-started-meson-build.md)

See those docs for more details about `nnstreamer` internals.

Currently supported EyePop platforms:
* Ubuntu 22.04 on `amd64` (fully supported)
* Ubuntu 22.04 on `aarch64` (TensorFlow Lite only)

## Development Environment Setup

For preexisting Docker images with all dependencies installed, see the
[Docker setup instructions][docker-setup]. The corresponding Dockerfiles can be found
[here][dockerfiles].

To develop on your host machine, install the following required packages through your system package
manager (instructions provided for `apt`). For EyePop-specific dependencies, add the EyePop package
repo to your sources list before proceeding:

```sh
sudo echo "deb [trusted=yes] http://repo.dev.eyepop.xyz.s3.us-east-1.amazonaws.com/ stable main" \
    | sudo tee /etc/apt/sources.list.d/eyepop.dev.list > /dev/null
```

`apt` instructions for installing everything below (leave out `eyepop-torchvision` on `aarch64`):

```sh
sudo apt install build-essential debhelper devscripts gcc9 meson ninja-build libglib2.0-dev \
     libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libjson-glib-dev gstreamer1.0-tools \
     gstreamer1.0-plugins-good libgtest-dev libopencv-dev flex bison python3-dev libpaho-mqtt-dev \
     eyepop-onnxruntime eyepop-tflite eyepop-torchvision
```

Build tools:
* `bison`
* `build-essential`
* `debhelper`
* `devscripts`
* `gcc-9`
* `flex`
* `meson`
* `ninja-build`
* `python3-dev`

Build dependencies:
* `libglib2.0-dev`
* `libgstreamer1.0-dev`
* `libgstreamer-plugins-base1.0-dev`
* `libgtest-dev`
* `libjson-glib-dev`
* `libopencv-dev`
* `libpaho-mqtt-dev`

Runtime dependencies:
* `gstreamer1.0-tools`
* `gstreamer1.0-plugins-good`

EyePop-specific dependencies:
* `eyepop-onnxruntime`
* `eyepop-tflite`
* `eyepop-torchvision` (amd64 only)

## Build/install instructions

### Developing locally

`nnstreamer` uses `meson` to orchestrate the build. Use individual feature flags to control what you
build for a particular platform.

To build/test for Ubuntu 22.04 on `amd64` with support for ONNX, TensorFlow, and TensorFlow Lite:

```sh
meson -Dwerror=false -Donnxruntime-support=enabled -Dtf-support=enabled -Dcaffe2-support=disabled -Dpython3-support=disabled build/ -Denable-test=false

ninja -C build

ninja -C build test
```

To build/test for Ubuntu 22.04 on `aarch64` with support for TensorFlow Lite and ONNX only:

```sh
meson -Dwerror=false -Donnxruntime-support=enabled -Dtf-support=disabled -Dcaffe2-support=disabled -Dpython3-support=disabled build/ -Denable-test=false

ninja -C build

ninja -C build test
```

### Building installable packages locally

To build installable `.deb` packages locally, use `debuild`. Like `eyepop-ml-dev`, the EyePop build
of `nnstreamer` uses platform identifiers to customize how packages get built.

| Platform                       | CUDA Support? | Identifier                 |
|--------------------------------|---------------|----------------------------|
| Linux, `amd64`, Ubuntu 22.04   | yes           | `linux-amd64-jammy-cuda`   |
| Linux, `amd64`, Ubuntu 22.04   | no            | `linux-amd64-jammy`        |
| Linux, `aarch64`, Ubuntu 22.04 | no            | `linux-aarch64-jammy`      |

To build for Ubuntu 22.04 on `amd64` with CUDA:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-amd64-jammy-cuda
```

To build for Ubuntu 22.04 on `amd64` without CUDA:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-amd64-jammy
```

To build for Ubuntu 22.04 on `aarch64`:

```sh
DEB_BUILD_OPTIONS="nocheck notest" debuild -b -us -uc -Plinux-aarch64-jammy
```

This will generate a series of `.deb` packages in the repo's parent directory. To install them with
`apt`, run the following:

```sh
sudo apt install ../nnstreamer*.deb
```

### Notes on packaging strategy

See `eyepop-ml-dev`'s README.

## Notes on MSYS2 build for Windows on ARM

These instructions are experimental, subject to change, and will eventually be integrated with the
setup instructions above after testing and verification.

Note: MSYS2 uses your entire Windows username for `$HOME`, which may include spaces. To work around
this, I actually used `/home/tyler/workspace/install` as the base of the install path. Change paths
below as desired, but using the paths below will work since the MSYS2 environment apparently doesn't
have strict permissions. Need to investigate further.

Note: on my machine, running a `pacman` command often resulted in two instances. It was necessary to
kill the one with (usually) lower memory through the Windows task manager and/or run the command
again (ignoring errors from the first run). After running a command successfully, check the Task
Manager (or run `ps -e | grep 'pacman'`) for any processes to kill.

### Initial setup

* Install MSYS2: https://github.com/msys2/msys2-installer/releases/tag/2024-07-27
* Run the `MSYS2 CLANGARM64` application (`"C:\msys64\clangarm64.exe"`, if installed to the default
  path)

Update and install build tools (run this twice -- will close the window once for a system upgrade):

```sh
pacman -S \
  mingw-w64-clang-aarch64-clang \
  mingw-w64-clang-aarch64-cmake \
  mingw-w64-clang-aarch64-gtest \
  mingw-w64-clang-aarch64-make \
  mingw-w64-clang-aarch64-pkgconf \
  mingw-w64-clang-aarch64-python3.12
```

Install dependencies for nnstreamer (can be combined with above instructions -- just separating for
tracking):

```sh
pacman -S \
  mingw-w64-clang-aarch64-glib2 \
  mingw-w64-clang-aarch64-libffi
```

### Build gstreamer

Clone `gstreamer` from https://gitlab.freedesktop.org/gstreamer/gstreamer and check out the latest
tag (1.24.9).

Set up a Python virtual environment to install necessary tools. Note: installing CMake through `pip`
did not work for me, hence the system package above.

```sh
python3.12 -m venv .venv
pip3 install meson ninja
```

To build and install `gstreamer` (note the GST-specific install directory to keep it separate from
the experimental `nnstreamer` build):

```sh
meson setup builddir \
  --prefix=/home/tyler/workspace/install/gst \
  -Dauto_features=disabled -Dgstreamer:tools=enabled -Dgood=enabled -Ddevtools=enabled -Dgstreamer:check=enabled -Dbad=enabled -Dwebrtc=enabled -Dlibnice=enabled -Dgst-plugins-bad:dtls=enabled -Dgst-plugins-bad:sctp=enabled -Dgst-plugins-bad:srtp=enabled -Dgst-plugins-good:rtpmanager=enabled -Dgst-plugins-good:autodetect=enabled -Dgst-plugins-base:videotestsrc=enabled -Dgst-plugins-base:videoconvertscale=enabled -Dgst-plugins-base:app=enabled -Dgst-plugins-base:audiotestsrc=enabled -Dgst-plugins-base:overlaycomposition=enabled -Dgst-plugins-bad:videoparsers=enabled -Dgst-plugins-good:flv=enabled -Dgst-plugins-bad:rtmp=enabled -Dugly=enabled -Dgpl=enabled -Dgst-plugins-ugly:x264=enabled -Dgst-plugins-bad:openh264=enabled -Dlibav=enabled -Dgst-plugins-base:pango=enabled -Dgst-plugins-bad:curl=enabled -Dgst-plugins-good:isomp4=enabled
```

```sh
meson compile -C builddir
```

```sh
meson install -C builddir
```

### tflite2 notes

My preference would have been to build tflite from `eyepop-ml-dev`, but compilation is currently
blocked by this MSYS2 package issue: https://github.com/msys2/MINGW-packages/issues/22160

### Install onnxruntime

Download v.1.19.2 of the CPU-accelerated ONNX runtime from
https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2

Unpack it to your desired install directory (in my case, `/home/tyler/workspace/install/onnx`).
There should be `lib/` and `include/` directories with no subdirectories, just files.

Create a `pkgconfig/` directory in the `lib/` folder and add the following `libonnxruntime.pc` file.
Again, adjust paths as necessary.

```
prefix=C:/msys64/home/tyler/workspace/install/onnx
includedir=${prefix}/include
libdir=${prefix}/lib

Name: ONNX runtime
Description: ONNX runtime libraries and headers
Version: 1.19.2
Libs: -L${libdir} -lonnxruntime -lonnxruntime_providers_shared
Cflags: -I${includedir}
```

### Build nnstreamer

Clone `eyepop-nnstreamer` from https://github.com/eyepop-ai/eyepop-nnstreamer and check out the
`tyler/msys2` branch.

Again, set up a Python virtual environment to install necessary tools. Note: installing CMake
through `pip` did not work for me, hence the system package above.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip3 install meson ninja
```

To build `nnstreamer` (which should pick up the ONNX runtime):

```sh
meson setup builddir \
  --prefix=/home/tyler/workspace/install/nnstreamer \
  -Dwerror=false -Denable-test=false -Denable-nnstreamer-check=false \
  -Dpkg_config_path=/home/tyler/workspace/install/gst/lib/pkgconfig:/home/tyler/workspace/install/onnx/lib/pkgconfig
```

```sh
ninja -C builddir
```

### Next steps

* Build `eyepop-pipeline`
* Run an ONNX model
* Fix all the warnings I introduced (e.g. duplicate definitions of `NNS_API`)
* Add proper error handling for `char*` to `wchar_t*` conversions
* Properly handle conditional build options for Linux and Windows (where I've just overwritten the
  defaults to hack together the Windows build)
* Make `nnstreamer` tests work
* Make this work in CI
* Figure out the weird `pacman` behavior
* Properly document the Windows build process and current status

It's likely that at this point, we'll discover several runtime issues with the nnstreamer build
and/or environment setup. But the build works on my machine! (for now)

[docker-setup]: https://app.gitbook.com/o/lp5TZAZIwu5jXdzZth9T/s/0fWYYHCrIOcShgRMiWhf/readme/howtos/running-as-docker-container
[dockerfiles]: https://github.com/eyepop-ai/eyepop-docker-images
