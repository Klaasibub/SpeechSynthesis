#!/bin/bash
./ngrok http 4567 &
python scripts/endless.py
