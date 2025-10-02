#!/bin/bash
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MINIMAL VERSION - Simple SSH daemon startup

# Add any provided SSH public key to authorized_keys
if [ -n "$SSH_PUBLIC_KEY" ]; then
    echo "$SSH_PUBLIC_KEY" >> $HOME/.ssh/authorized_keys
fi

# Start SSH daemon using system sshd
/usr/sbin/sshd -D -e -f $HOME/.ssh/sshd_config

