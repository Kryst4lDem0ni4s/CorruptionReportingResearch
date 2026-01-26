# Synthetic Coordinated Attacks

## Description

Generated synthetic attack scenarios for testing coordination detection.

## Statistics

- Attack groups: 10
- Total submissions: 58
- Group sizes: 3-10
- Patterns: linguistic_similarity, temporal_clustering, style_consistency, coordinated_timing, identical_evidence

## Pattern Distribution

- linguistic_similarity: 3
- style_consistency: 4
- temporal_clustering: 1
- identical_evidence: 2

## Data Format

```json
{
  "id": "attack_001",
  "group_size": 5,
  "pattern": "linguistic_similarity",
  "target": "John Smith",
  "submissions": [...],
  "metrics": {
    "similarity_score": 0.85,
    "temporal_window_hours": 2.5
  }
}

# Command-line usage
# python evaluation/datasets/generate_synthetic.py --num-groups 20 --with-defense
