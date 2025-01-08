#!/bin/bash

source venv/bin/activate

mypy --follow-imports=skip $1 --strict