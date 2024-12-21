import traceback

import boto3
import os
import tempfile
from pathlib import Path
from botocore.exceptions import ClientError

import logging

# make this region dynamic later
s3_client = boto3.client('s3', region_name="us-west-2")
logger = logging.getLogger(__name__)

def check_s3_object_exists(bucket_name, object_key):
    """
    Check if an object exists in an S3 bucket.

    This function performs a HEAD request on the S3 object to verify its existence
    without downloading the object content. It handles the AWS ClientError exception
    to determine if the object exists.

    Args:
        bucket_name (str): The name of the S3 bucket to check.
        object_key (str): The key (path) of the object within the bucket.

    Returns:
        bool: True if the object exists, False if it doesn't.

    Raises:
        botocore.exceptions.ClientError: If there's an error other than 404 (like permissions issues,
            invalid bucket name, network problems, etc.)
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise

def download_s3_file_in_chunks(bucket_name, object_key, chunk_size=1024*1024):  # 1MB chunks
    """
    Download a file from S3 in chunks and save to temp directory.
    TODO: This is downloading the file in sequence. We will make this in parallel in next iteration
    
    Args:
        bucket_name (str): The S3 bucket name
        object_key (str): The S3 object key (file path)
        chunk_size (int): Size of chunks to download (default 1MB)
        
    Returns:
        str: Path to the downloaded file in temp directory
    """
    
    try:
        # Get object details
        logger.info(f"Bucket name: {bucket_name}, Object key: {object_key}")
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']
        
        # Create temp file with same extension as original
        file_extension = Path(object_key).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file_path = temp_file.name
        
        # Get the object
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        downloaded = 0
        with open(temp_file_path, 'wb') as f:
            logger.info(f"Downloading {object_key} to {temp_file_path}")
            
            # Download and write chunks
            for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Calculate and display progress
                progress = (downloaded / file_size) * 100
                logger.info(f"Progress: {progress:.2f}% ({downloaded}/{file_size} bytes)")

        logger.info(f"Download completed: {temp_file_path}")
        return temp_file_path
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.info(f"The object {object_key} does not exist in bucket {bucket_name}")
        elif error_code == 'NoSuchBucket':
            logger.info(f"The bucket {bucket_name} does not exist")
        else:
            logger.info(f"Error downloading object: {traceback.format_exc()} {e}")
            
        # Clean up temp file if it exists
        cleanup_temp_file(temp_file_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()} {e}")
        # Clean up temp file if it exists
        cleanup_temp_file(temp_file_path)
        raise

def cleanup_temp_file(temp_file_path):
    """
    Clean up the temporary file when no longer needed
    
    Args:
        temp_file_path (str): Path to the temporary file
    """
    try:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Temporary file removed: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error removing temporary file: {traceback.format_exc()} {e}")