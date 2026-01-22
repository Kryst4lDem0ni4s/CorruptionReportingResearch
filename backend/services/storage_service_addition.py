    def save_evidence_file(
        self,
        submission_id: str,
        filename: str,
        content: bytes,
        evidence_type: str
    ) -> Path:
        """
        Save evidence file content to storage.
        
        Args:
            submission_id: Submission identifier
            filename: Original filename
            content: File content bytes
            evidence_type: Type of evidence
            
        Returns:
            Path: Path to saved file
        """
        try:
            # Get evidence path with sharding
            safe_filename = self._sanitize_filename(filename)
            file_path = self.get_evidence_path(submission_id, safe_filename)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.debug(f"Saved evidence file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save evidence file: {e}")
            raise IOError(f"Failed to save evidence file: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re
        # Remove path components
        filename = Path(filename).name
        # Replace unsafe characters
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        # Limit length
        if len(safe_name) > 255:
            name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name[:250] + ('.' + ext if ext else '')
        return safe_name or 'unnamed_file'
