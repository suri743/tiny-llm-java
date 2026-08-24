#!/bin/bash
# Run this ONCE manually from your local machine to create the OIDC role for GitHub Actions.
# Prerequisites: aws cli configured with admin access to account 652386518241

set -e

ACCOUNT_ID="652386518241"
REGION="ap-south-1"
ROLE_NAME="llm-training-github-actions"

echo "Creating OIDC provider for GitHub Actions..."
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
  --region $REGION 2>/dev/null || echo "OIDC provider already exists"

OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"

echo "Creating IAM role..."
aws iam create-role \
  --role-name $ROLE_NAME \
  --assume-role-policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Principal\": { \"Federated\": \"${OIDC_ARN}\" },
      \"Action\": \"sts:AssumeRoleWithWebIdentity\",
      \"Condition\": {
        \"StringEquals\": {
          \"token.actions.githubusercontent.com:aud\": \"sts.amazonaws.com\"
        },
        \"StringLike\": {
          \"token.actions.githubusercontent.com:sub\": \"repo:suri743/tiny-llm-java:*\"
        }
      }
    }]
  }" 2>/dev/null || echo "Role already exists"

echo "Attaching AdministratorAccess policy..."
aws iam attach-role-policy \
  --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

echo ""
echo "Done! Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
