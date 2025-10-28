#!/bin/bash

# Setup script to create directory paths from .env.template if they don't exist

set -e  # Exit on any error

echo "Setting up directory structure from .env.template..."

# Check if .env.template exists
if [ ! -f ".env.template" ]; then
    echo "ERROR: .env.template file not found!"
    exit 1
fi

# Extract directory paths from .env.template
DIRS=$(grep -E '^(DATA_DIR|CACHE_DIR|PROCESSED_DIR|INDEX_DIR|QUERY_DIR|RESPONSE_DIR)=' .env.template | cut -d'=' -f2 | sed 's/#.*//' | sed 's/[[:space:]]*$//')

echo "Creating directories from .env.template..."

# Create each directory
for DIR in $DIRS; do
    if [ ! -d "$DIR" ]; then
        mkdir -p "$DIR"
        echo "✅ Created: $DIR"
    else
        echo "📁 Already exists: $DIR"
    fi
done

echo ""
echo "🎉 Directory setup complete!"
echo ""
echo "Directory structure:"
for DIR in $DIRS; do
    echo "  - $DIR"
done