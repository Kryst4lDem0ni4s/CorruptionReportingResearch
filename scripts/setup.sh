#!/bin/bash

# Setup script for corruption reporting prototype

echo "Setting up data directory structure..."

# Base data directory
DATA_DIR="backend/data"

# Create directories
mkdir -p "$DATA_DIR/submissions"
mkdir -p "$DATA_DIR/evidence/2026/"{01..12}
mkdir -p "$DATA_DIR/reports"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/reports_archive"

# Create .gitkeep files
find "$DATA_DIR" -type d -exec touch {}/.gitkeep \;

# Create initial JSON files
cat > "$DATA_DIR/chain.json" << 'EOF'
{
  "genesis_block": {
    "block_number": 0,
    "timestamp": "2026-01-13T21:23:00+05:30",
    "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    "data": "Genesis Block",
    "hash": null
  },
  "blocks": [],
  "created_at": "2026-01-13T21:23:00+05:30"
}
EOF

cat > "$DATA_DIR/validators.json" << 'EOF'
{
  "validators": [],
  "created_at": "2026-01-13T21:23:00+05:30",
  "last_updated": "2026-01-13T21:23:00+05:30"
}
EOF

cat > "$DATA_DIR/index.json" << 'EOF'
{
  "submissions": {},
  "created_at": "2026-01-13T21:23:00+05:30",
  "last_updated": "2026-01-13T21:23:00+05:30"
}
EOF

echo "✓ Data directory structure created successfully!"
