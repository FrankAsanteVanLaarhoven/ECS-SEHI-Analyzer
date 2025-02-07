#!/bin/bash

# Configuration
REPO="ecs_sehi-analysis-dashboard"
USERNAME="FrankAsanteVanLaarhoven"
EMAIL="info@franvanlaarhoven.cco.uk"
BRANCH="main"

# Initialize git if not already initialized
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# Configure git
git config user.name "$USERNAME"
git config user.email "$EMAIL"

# Add remote if not already added
if ! git remote | grep -q origin; then
    git remote add origin "https://github.com/$USERNAME/$REPO.git"
fi

# Stage all changes
git add .

# Commit changes
git commit -m "Update deployment configuration and dependencies"

# Push to remote
git push -u origin $BRANCH