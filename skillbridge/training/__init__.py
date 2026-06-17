"""Training resource registry (skill-gap recommender).

See:
  docs/training-registry-schema.md      -- YAML structure + validation rules
  docs/training-providers-allowlist.md  -- which providers we trust and why
  data/training_registry.yaml           -- v1 seed (pending URL verification)

Loaded once at server startup via TrainingRegistry.from_yaml().
Handler integration is gated behind TRAINING_REGISTRY_ENABLED env flag.
"""
