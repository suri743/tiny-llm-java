output "instance_id" {
  value = aws_instance.training.id
}

output "instance_public_ip" {
  value = aws_instance.training.public_ip
}

output "training_bucket" {
  value = aws_s3_bucket.llm_training.bucket
}
