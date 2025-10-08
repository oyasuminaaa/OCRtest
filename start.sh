#!/bin/sh

# Start Gunicorn with a 120-second timeout
gunicorn --bind 0.0.0.0:${PORT} --timeout 120 backend.app:app
