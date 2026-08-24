############################
# S3 — JAR storage + TF state
############################

resource "aws_s3_bucket" "llm_training" {
  bucket        = "instyte-llm-training"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "llm_training" {
  bucket = aws_s3_bucket.llm_training.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

############################
# IAM — EC2 can read from S3
############################

resource "aws_iam_role" "ec2_training" {
  name = "llm-training-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "s3_read" {
  name = "llm-training-s3-read"
  role = aws_iam_role.ec2_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.llm_training.arn,
        "${aws_s3_bucket.llm_training.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "ec2_training" {
  name = "llm-training-ec2-profile"
  role = aws_iam_role.ec2_training.name
}

############################
# VPC + Subnet
############################

resource "aws_vpc" "training" {
  cidr_block           = "10.99.0.0/16"
  enable_dns_hostnames = true

  tags = { Name = "llm-training-vpc" }
}

resource "aws_internet_gateway" "training" {
  vpc_id = aws_vpc.training.id
  tags   = { Name = "llm-training-igw" }
}

resource "aws_subnet" "training" {
  vpc_id                  = aws_vpc.training.id
  cidr_block              = "10.99.1.0/24"
  availability_zone       = "ap-south-1a"
  map_public_ip_on_launch = true

  tags = { Name = "llm-training-subnet" }
}

resource "aws_route_table" "training" {
  vpc_id = aws_vpc.training.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.training.id
  }

  tags = { Name = "llm-training-rt" }
}

resource "aws_route_table_association" "training" {
  subnet_id      = aws_subnet.training.id
  route_table_id = aws_route_table.training.id
}

############################
# Security Group — outbound only
############################

resource "aws_security_group" "training" {
  name        = "llm-training-sg"
  description = "LLM training EC2 - outbound only"
  vpc_id      = aws_vpc.training.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "llm-training-sg" }
}

############################
# EC2 Instance
############################

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "training" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t3.medium"
  subnet_id              = aws_subnet.training.id
  iam_instance_profile   = aws_iam_instance_profile.ec2_training.name
  vpc_security_group_ids = [aws_security_group.training.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e

    dnf install -y java-21-amazon-corretto-headless aws-cli

    mkdir -p /opt/llm-training
    aws s3 cp s3://instyte-llm-training/tiny-llm.jar /opt/llm-training/tiny-llm.jar

    cd /opt/llm-training
    nohup java -Xmx3g -jar tiny-llm.jar > /opt/llm-training/training.log 2>&1 &

    echo "Training started"
  EOF

  tags = { Name = "llm-training" }
}
