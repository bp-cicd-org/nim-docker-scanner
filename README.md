# NIM Docker Scanner

A reusable GitHub Actions workflow for scanning Docker images and hosted NIM endpoints in NVIDIA projects.

## Features

- **Docker Image Scanning**: Scans workflows, Dockerfiles, and docker-compose files for Docker image references
- **Latest Tag Resolution**: Resolves `latest` tags to actual digests via `docker pull`
- **Hosted NIM Detection**: Identifies hosted NIM endpoints (integrate.api.nvidia.com, ai.api.nvidia.com)
- **NVCF Function ID Matching**: Matches hosted NIMs with NVCF function IDs from configuration
- **JSON Reports**: Generates detailed JSON reports as GitHub Actions artifacts

## Usage

### As a Reusable Workflow

Add to your workflow file:

```yaml
jobs:
  scan:
    uses: bp-cicd-org/nim-docker-scanner/.github/workflows/scan.yml@main
    with:
      exclude-dirs: 'external,vendor'
      resolve-latest: true
      scan-docker-images: true
      scan-hosted-nim: true
    secrets:
      NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
      NVCF_CONFIG_ON_GITHUB_TOKEN: ${{ secrets.NVCF_CONFIG_ON_GITHUB_TOKEN }}
```

### After Another Job

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy application
        run: echo "Deploying..."

  scan:
    needs: deploy
    if: always()
    uses: bp-cicd-org/nim-docker-scanner/.github/workflows/scan.yml@main
    with:
      exclude-dirs: 'external'
    secrets:
      NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
      NVCF_CONFIG_ON_GITHUB_TOKEN: ${{ secrets.NVCF_CONFIG_ON_GITHUB_TOKEN }}
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `exclude-dirs` | Comma-separated list of directories to exclude | No | `''` |
| `resolve-latest` | Resolve latest Docker tags to digests | No | `true` |
| `scan-docker-images` | Enable Docker image scanning | No | `true` |
| `scan-hosted-nim` | Enable hosted NIM endpoint scanning | No | `true` |
| `nvcf-config-repo` | NVCF API Gateway config repository URL | No | `https://github.com/bp-cicd-org/nvcf-api-gateway-config-prd.git` |
| `nvcf-config-file` | NVCF config YAML file name | No | `nvcf-api-gateway-prd.yaml` |

## Secrets

| Secret | Description | Required |
|--------|-------------|----------|
| `NVIDIA_API_KEY` | NVIDIA API key for nvcr.io access (for resolving latest tags) | No |
| `NVCF_CONFIG_ON_GITHUB_TOKEN` | GitHub token for accessing NVCF config repository | No |

## Output Artifacts

### Docker Image Report

JSON file containing:
- `metadata`: Scan timestamp, repository info, commit SHA
- `actions_workflow_images`: Images used in GitHub Actions workflows
- `non_actions_images`: Images found in repo but not used in workflows

Example:
```json
{
  "metadata": {
    "scan_time": "2025-01-16T10:30:00Z",
    "repo_name": "org/repo",
    "commit": "abc123..."
  },
  "actions_workflow_images": [
    {
      "image": "nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
      "tag": "1.10.0",
      "source_file": ".github/workflows/deploy.yml",
      "source_line": 122,
      "is_local_build": false
    }
  ],
  "non_actions_images": [...]
}
```

### Hosted NIM Report

JSON file containing:
- `metadata`: Scan timestamp, repository info
- `actions_hosted_nim`: Hosted NIMs used in workflows
- `non_actions_hosted_nim`: Hosted NIMs in repo code
- `summary`: Match statistics

Example:
```json
{
  "metadata": {
    "scan_time": "2025-01-16T10:30:00Z",
    "repo_name": "org/repo"
  },
  "actions_hosted_nim": [],
  "non_actions_hosted_nim": [
    {
      "model": "nvidia/nvidia-nemotron-nano-9b-v2",
      "base_url": "https://integrate.api.nvidia.com/v1",
      "source_file": "config/config.yaml",
      "function_id": "abc-123-def",
      "matched": true
    }
  ],
  "summary": {
    "total_actions_refs": 0,
    "matched_actions_refs": 0,
    "total_repo_refs": 2,
    "matched_repo_refs": 2
  }
}
```

## Repository Access

This repository must be accessible to calling workflows. For private repositories:

1. Go to **Settings > Actions > General**
2. Under "Access", select "Accessible from repositories in the organization"

## License

See [LICENSE](LICENSE) for details.
