
# Corruption Reporting System - API Specification
**Version:** 1.0.0-mvp
**Base URL:** `/api/v1`

## Overview
This document defines the strict API contract for the Corruption Reporting System. All backend implementations and frontend integrations must adhere effectively to this specification.

## 1. Health & Monitoring

### 1.1 System Health
**Endpoint:** `GET /api/v1/health`
**Description:** Quick status check for system components.
**Response Content-Type:** `application/json`

**Response Schema:**
```json
{
  "status": "string",          // "healthy" | "degraded" | "unhealthy"
  "timestamp": "datetime",     // ISO 8601 UTC
  "checks": {                  // Component status
    "storage_readable": true,
    "storage_writable": true,
    "hash_chain_valid": true,
    "crypto_operational": true,
    "models_loadable": true,
    "memory_ok": true,
    "memory_percent": 45.2,    // Float or null
    "disk_ok": true,
    "disk_percent": 12.5       // Float or null
  },
  "uptime_seconds": 3600.5,
  "version": "1.0.0-mvp"
}
```
**Status Codes:**
- `200 OK`: System healthy or degraded.
- `503 Service Unavailable`: System unhealthy (critical failure).

### 1.2 Detailed Health
**Endpoint:** `GET /api/v1/health/detailed`
**Description:** Extended diagnostic information.
**Response Schema:** `Dict` (Open structure for debug info)

### 1.3 Metrics
**Endpoint:** `GET /api/v1/metrics`
**Description:** Prometheus-formatted metrics.
**Response Content-Type:** `text/plain`

---

## 2. Submissions

### 2.1 Submit Evidence
**Endpoint:** `POST /api/v1/submissions`
**Content-Type:** `multipart/form-data`

**Form Fields:**
- `file` (File): Evidence file (image/jpeg, img/png, audio/wav, video/mp4). Max 100MB.
- `description` (String, Optional): Text narrative (max 5000 chars).
- `location` (String, Optional): Location string.
- `incident_date` (String, Optional): ISO 8601 Date string.

**Response Schema (Success):**
```json
{
  "submission_id": "uuid-string",
  "pseudonym": "string",
  "evidence_hash": "sha256-string",
  "timestamp": "iso-date",
  "status": "pending"
}
```
**Status Codes:**
- `201 Created`: Submission accepted.
- `400 Bad Request`: Invalid file type, file too large, or validation error.
- `429 Too Many Requests`: Rate limit exceeded.

### 2.2 Get Submission Status
**Endpoint:** `GET /api/v1/submissions/{submission_id}`
**Response Schema:**
```json
{
  "submission_id": "uuid-string",
  "status": "completed",
  "credibility": {
    "deepfake_score": 0.1,    // 0.0-1.0 (Lower is real)
    "consistency_score": 0.9, // 0.0-1.0
    "final_score": 0.85,      // 0.0-1.0 (Higher is authentic)
    "confidence_interval": [0.8, 0.9]
  },
  "coordination": { ... },
  "consensus": { ... }
}
```
**Status Codes:**
- `200 OK`: Found.
- `404 Not Found`: Invalid ID.

---

## 3. Global Error Format
All 4xx and 5xx responses (except 429) follow this format:
```json
{
  "error": "ErrorType",
  "message": "Human readable message",
  "details": { ... },   // Optional context
  "timestamp": "iso-date"
}
```
