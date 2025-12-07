#!/bin/sh
set -e

host="$1"
port="$2"
shift 2

echo "Waiting for $host:$port..."

while ! nc -z "$host" "$port"; do
  echo "  ."
  sleep 1
done

echo "Database at $host:$port is available — running command"
exec "$@"
