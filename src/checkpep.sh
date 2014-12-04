#!/usr/bin/env bash

# Check that source code conforms to PEP8.
if pep8 --show-source *.py; then
	echo "No style errors found. You're so hip!"
else
	echo "Some style errors detected."
fi
echo ""

