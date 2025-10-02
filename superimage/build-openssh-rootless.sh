#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ROOTLESS VERSION - Builds OpenSSH in user space

set -e
set -x

# Use conda environment for build tools
eval "$(conda shell.bash hook)"

umask 0077
# Install to user directory instead of system
prefix="$HOME/opt/openssh"
top="$(pwd)/openssh_build"
root="$top/root"
build="$top/build"
export CPPFLAGS="-I$root/include -L."
rm -rf "$root" "$build" "$top"
mkdir -p "$root" "$build" "$top"
cd "$top"
mkdir dist
cd dist

# Download sources
wget https://www.zlib.net/fossils/zlib-1.2.11.tar.gz
wget https://github.com/openssl/openssl/releases/download/OpenSSL_1_0_2n/openssl-1.0.2n.tar.gz
wget https://cdn.openbsd.org/pub/OpenBSD/OpenSSH/portable/openssh-7.4p1.tar.gz
cd ..

# Build zlib
gzip -dc dist/zlib-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/zlib-*
./configure --prefix="$root" --static
make
make install
cd "$top"

# Build OpenSSL
gzip -dc dist/openssl-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/openssl-*
./config --prefix="$root" no-shared
make
make install
cd "$top"

# Build OpenSSH
gzip -dc dist/openssh-*.tar.gz |(cd "$build" && tar xf -)
cd "$build"/openssh-*
cp -p "$root"/lib/*.a .

# Configure for user installation with user privilege separation
./configure \
    --prefix="$prefix" \
    --with-privsep-user=$(whoami) \
    --with-privsep-path="$prefix/var/empty" \
    --disable-strip \
    --with-ssl-dir="$root"

# Remove locked password prefix check (not needed for user space)
sed -i '/LOCKED_PASSWD_PREFIX/d' config.h

make -j $(nproc)
make install-nokeys

# Create privilege separation directory
mkdir -p "$prefix/var/empty"

cd "$HOME"
rm -rf "$top"

echo "OpenSSH built successfully in $prefix"

