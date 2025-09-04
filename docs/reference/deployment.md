# GitHub Actions Deployment Workflow Visualization

This document visualizes the deployment workflow logic you described, including triggers, input-based branching, and environment variable assignments.

## Mermaid Diagram

```
                          +------------------------+
                          |  Workflow Event Fired  |
                          +-----------+------------+
                                      |
                    +-----------------+------------------+
                    |                                    |
             (release: published)               (workflow_dispatch)
                    |                                    |
                    v                                    v
         +-------------------------+          +---------------------------+
         | deploy-production job   |          | Evaluate inputs           |
         | (automatic prod deploy) |          | (production_tag? etc.)    |
         +------------+------------+          +-------------+-------------+
                      |                                     |
                      |                                     |
                      |                             production_tag provided?
                      |                                     |
                      |                    +----------------+----------------+
                      |                    |                                 |
                      |                   YES                               NO
                      |                    |                                 |
                      |                    v                                 v
                      |        +---------------------------+      +---------------------------+
                      |        | production-rerun job      |      | deploy job (non-prod)     |
                      |        | manual prod redeploy      |      | qa or staging             |
                      |        +-------------+-------------+      +--------------+------------+
                      |                      |                                |
                      |                      |                                |
                      |                Checkout tag                            |
                      |                (actions/checkout@v5)                  |
                      |                      |                                |
                      |                Derive OCR_VERSION                     |
                      |                - RAW_TAG from                         |
                      |                  github.event.inputs.production_tag   |
                      |                - Strip refs/tags/ and leading v       |
                      |                - Validate SemVer                      |
                      |                Set:                                   |
                      |                  OCR_VERSION                          |
                      |                  OCR_ENVIRONMENT=production           |
                      |                      |                                |
                      |                      v                                v
                      |               Run deploy (prod)          Set OCR_ENVIRONMENT=qa|staging
                      |               - REGION_ARGS="--all..."                |
                      |               - env-file: prod                       |
                      |                                                     Build region args:
                      |                                               IF all_region_ids == true
                      |                                                 -> --all-region-ids
                      |                                                   (ignore region_id)
                      |                                               ELSE IF region_id non-empty
                      |                                                 -> --region-id <value>
                      |                                               ELSE -> FAIL (exit 1)
                      |                                                     |
                      |                                                     v
                      |                                              Optional wipe:
                      |                                                if wipe == true -> --wipe
                      |                                                     |
                      |                                                     v
                      |                                              Run deploy (qa/staging)
                      |                                              - env-file: ocr-coiled-s3.env
                      |                                              - OCR_VERSION NOT set here
                      |                                                     |
                      |                                                     v
                      |                                             (End non-prod job)
                      |
                      v
   +----------------------------------+
   | deploy-production job (release) |
   +-----------------+---------------+
                     |
              actions/checkout
                     |
              Derive OCR_VERSION
              - RAW_TAG from github.event.release.tag_name
              - Strip refs/tags/ and leading v
              - Validate SemVer
              Set:
                OCR_VERSION
                OCR_ENVIRONMENT=production
                     |
                     v
              Run production deploy
              - REGION_ARGS="--all-region-ids"
              - env-file: ocr-coiled-s3-prod.env
                     |
                     v
              (End production deploy)

FAIL CONDITIONS:
  - Non-prod deploy: neither all_region_ids nor region_id provided
  - SemVer validation fails for production_tag or release tag

SUMMARY OF VARIABLES:
  - OCR_ENVIRONMENT:
       deploy (manual): qa | staging
       deploy-production (release): production
       production-rerun: production
  - OCR_VERSION:
       Only set in production jobs (release / production-rerun)

REGION ARG LOGIC (non-prod):
  all_region_ids == true  -> use --all-region-ids (ignore region_id if given)
  else if region_id set   -> use --region-id <region_id>
  else                    -> fail

PRODUCTION ALWAYS:
  --all-region-ids

```

## Logical Summary

- Triggers:

  - release (published) -> production deploy (deploy-production job)
  - workflow_dispatch -> either:
    - production rerun if production_tag is provided
    - non-prod deploy (qa or staging) if production_tag is empty

- workflow_dispatch inputs:

  - environment (qa | staging) — ignored if production_tag is provided
  - production_tag — SemVer tag (with or without leading v); if set, triggers production rerun
  - region_id — used only when all_region_ids == false
  - all_region_ids — boolean toggle for region fan-out
  - wipe — boolean (assumed to influence job behavior, not branching)

- Environment variables:

  - Production (release or rerun):
    - OCR_VERSION = normalized SemVer (strip leading v)
    - OCR_ENVIRONMENT = production
  - Non-prod:
    - OCR_ENVIRONMENT = qa or staging

- Branching:
  1. Event == release -> deploy-production
  2. Event == workflow_dispatch:
     a. production_tag set -> production-rerun
     b. production_tag not set -> deploy (qa or staging) + region logic
