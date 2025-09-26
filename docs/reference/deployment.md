# GitHub Actions Deployment Workflow Visualization

This document visualizes the deployment workflow logic used to deploy OCR including triggers, input-based branching, and environment variable assignments.

## Mermaid Diagram

```mermaid
graph TB
    %% Trigger Events
    Start([Workflow Triggers])
    Push[Push to main]
    PR[Pull Request to main]
    Release[Release Published]
    Manual[Manual Workflow Dispatch]

    Start --> Push
    Start --> PR
    Start --> Release
    Start --> Manual

    %% Coiled Software Environment Job
    CoiledSoftware[["ocr-coiled-software<br/>Create Coiled Environment<br/>Output: name"]]

    Push --> CoiledSoftware
    PR --> CoiledSoftware
    Release --> CoiledSoftware
    Manual --> CoiledSoftware

    %% Conditional Jobs
    QA_PR{{"qa-pr<br/>IF: PR with e2e/QA labels<br/>Environment: qa"}}
    Staging_Main{{"staging-main<br/>IF: Push to main<br/>Environment: staging"}}
    Manual_Deploy{{"manual<br/>IF: Manual & no prod tag<br/>Environment: qa/staging"}}
    Production{{"production<br/>IF: Release published<br/>Environment: production"}}
    Production_Rerun{{"production-rerun<br/>IF: Manual & prod tag<br/>Environment: production"}}

    %% Job Dependencies and Conditions
    CoiledSoftware --> QA_PR
    CoiledSoftware --> Staging_Main
    CoiledSoftware --> Manual_Deploy
    CoiledSoftware --> Production
    CoiledSoftware --> Production_Rerun

    PR --> |"Has e2e or QA/QC label"| QA_PR
    Push --> |"Branch = main"| Staging_Main
    Manual --> |"production_tag = empty"| Manual_Deploy
    Release --> Production
    Manual --> |"production_tag != empty"| Production_Rerun

    %% Job Details
    QA_PR_Details[["QA Deploy<br/>• Regions: y2_x5-x7<br/>• Wipe: true<br/>• URL: ocr.qa.carbonplan.org"]]
    Staging_Main_Details[["Staging Deploy<br/>• Regions: Multiple specified<br/>• Wipe: true<br/>• URL: ocr.staging.carbonplan.org"]]
    Manual_Deploy_Details[["Manual Deploy<br/>• Regions: User choice<br/>• Wipe: User choice<br/>• URL: Based on environment"]]
    Production_Details[["Production Deploy<br/>• Regions: All<br/>• Wipe: false<br/>• URL: ocr.carbonplan.org"]]
    Production_Rerun_Details[["Production Redeploy<br/>• Regions: All<br/>• Wipe: false<br/>• URL: ocr.carbonplan.org"]]

    QA_PR --> QA_PR_Details
    Staging_Main --> Staging_Main_Details
    Manual_Deploy --> Manual_Deploy_Details
    Production --> Production_Details
    Production_Rerun --> Production_Rerun_Details

    %% Styling
    classDef trigger fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef job fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef conditional fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef deploy fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px

    class Start,Push,PR,Release,Manual trigger
    class CoiledSoftware job
    class QA_PR,Staging_Main,Manual_Deploy,Production,Production_Rerun conditional
    class QA_PR_Details,Staging_Main_Details,Manual_Deploy_Details,Production_Details,Production_Rerun_Details deploy

```
