$ROOT = "E:\Liver wala thing"
$RAW = "$ROOT\data\raw"
$PLAN = "$ROOT\data\download_plan.json"

Write-Host ""
Write-Host "========================================"
Write-Host "HCC-TACE-SEG SMART DATASET DOWNLOAD"
Write-Host "========================================"
Write-Host ""

$plan = Get-Content $PLAN -Raw | ConvertFrom-Json

$total = $plan.Count
$count = 0

foreach ($item in $plan) {

    $count++
    $patient = $item.patient

    Write-Host ""
    Write-Host "[$count/$total] $patient"
    Write-Host "----------------------------------------"

    # ==========================================
    # 1. DOWNLOAD SEG
    # ==========================================

    $segPath = $item.seg.path

    Write-Host "Downloading SEG..."

    hf download MedOtter/HCC-TACE-Seg `
        --repo-type dataset `
        --local-dir $RAW `
        --include "$segPath/*"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: SEG download failed for $patient"
        continue
    }

    # Find downloaded SEG DICOM
    $segDir = Join-Path $RAW $segPath
    $segFiles = Get-ChildItem $segDir -Filter "*.dcm" -File -ErrorAction SilentlyContinue

    if ($segFiles.Count -eq 0) {
        Write-Host "ERROR: SEG DICOM not found for $patient"
        continue
    }

    $segFile = $segFiles[0].FullName

    Write-Host "SEG found:"
    Write-Host $segFile

    # ==========================================
    # 2. READ REFERENCED CT SERIES
    # ==========================================

    Write-Host "Reading SEG -> CT reference..."

    $pythonCode = @"
import pydicom

p = r'''$segFile'''

ds = pydicom.dcmread(p, stop_before_pixels=True)

refs = []

if hasattr(ds, "ReferencedSeriesSequence"):
    for ref in ds.ReferencedSeriesSequence:
        if hasattr(ref, "SeriesInstanceUID"):
            refs.append(str(ref.SeriesInstanceUID))

if refs:
    print(refs[0])
else:
    print("NO_REFERENCE")
"@

    $refUid = $pythonCode | python

    $refUid = $refUid.Trim()

    if ($refUid -eq "NO_REFERENCE" -or [string]::IsNullOrWhiteSpace($refUid)) {
        Write-Host "ERROR: Could not find referenced CT series for $patient"
        continue
    }

    Write-Host "Referenced CT Series:"
    Write-Host $refUid

    # ==========================================
    # 3. FIND REFERENCED SERIES IN PLAN
    # ==========================================

    $ctMatch = $null

    foreach ($ct in $item.ct_candidates) {
        if ($ct.series_uid -eq $refUid) {
            $ctMatch = $ct
            break
        }
    }

    if ($null -eq $ctMatch) {
        Write-Host "ERROR: Referenced CT series not found in download plan."
        continue
    }

    Write-Host "CT series confirmed."
    Write-Host "CT slices: $($ctMatch.image_count)"
    Write-Host "CT path: $($ctMatch.path)"

    # ==========================================
    # 4. DOWNLOAD ONLY REFERENCED CT SERIES
    # ==========================================

    Write-Host ""
    Write-Host "Downloading referenced CT..."

    hf download MedOtter/HCC-TACE-Seg `
        --repo-type dataset `
        --local-dir $RAW `
        --include "$($ctMatch.path)/*"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: CT download failed for $patient"
        continue
    }

    Write-Host ""
    Write-Host "========================================"
    Write-Host "$patient COMPLETE"
    Write-Host "========================================"
}

Write-Host ""
Write-Host "========================================"
Write-Host "SMART DOWNLOAD FINISHED"
Write-Host "========================================"