Param(
  [string]$HostUrl = "http://localhost:11434",
  [string[]]$Models = @(),
  [int]$Runs = 3,
  [int]$Warmup = 1,
  [double]$TimeoutS = 600,
  [string]$Out = "",
  [string]$Prompt = "",
  [string]$PromptFile = "",
  [string]$KeepAlive = "5m",
  [int]$NumPredict = 256,
  [double]$Temperature = 0.0,
  [Nullable[double]]$TopP = $null,
  [Nullable[int]]$Seed = 42,
  [Nullable[int]]$NumCtx = $null,
  [string]$Stop = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function IsoNowLocal {
  return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

function NsToS([object]$ns) {
  if ($null -eq $ns) { return $null }
  try { return ([double]$ns) / 1000000000.0 } catch { return $null }
}

function SafeDiv([Nullable[double]]$n, [Nullable[double]]$d) {
  if ($null -eq $n -or $null -eq $d -or $d -eq 0) { return $null }
  return $n / $d
}

function Mean([double[]]$xs) {
  $arr = @($xs)
  if ($arr.Count -eq 0) { return $null }
  return ($arr | Measure-Object -Average).Average
}

function SampleStdev([double[]]$xs) {
  $arr = @($xs)
  if ($arr.Count -lt 2) { return $null }
  $m = (Mean $arr)
  $sum = 0.0
  foreach ($x in $arr) { $sum += [math]::Pow(($x - $m), 2) }
  return [math]::Sqrt($sum / ($arr.Count - 1))
}

function Percentile([double[]]$xs, [double]$p) {
  $arr = @($xs)
  if ($arr.Count -eq 0) { return $null }
  if ($arr.Count -eq 1) { return $arr[0] }
  $sorted = @($arr | Sort-Object)
  if ($p -le 0) { return $sorted[0] }
  if ($p -ge 100) { return $sorted[$sorted.Count - 1] }
  $k = ($sorted.Count - 1) * ($p / 100.0)
  $f = [math]::Floor($k)
  $c = [math]::Ceiling($k)
  if ($f -eq $c) { return $sorted[[int]$k] }
  $d0 = $sorted[$f] * ($c - $k)
  $d1 = $sorted[$c] * ($k - $f)
  return $d0 + $d1
}

function FmtFloat([Nullable[double]]$x, [int]$digits = 2) {
  if ($null -eq $x) { return "-" }
  return ([math]::Round($x, $digits)).ToString("F$digits")
}

function FmtInt([Nullable[int]]$x) {
  if ($null -eq $x) { return "-" }
  return "$x"
}

function MdEscape([string]$s) {
  return $s.Replace("|", "\|")
}

function TryGetProp([object]$obj, [string]$name) {
  if ($null -eq $obj) { return $null }
  $p = $obj.PSObject.Properties[$name]
  if ($null -eq $p) { return $null }
  return $p.Value
}

function HttpJson([string]$Url, [string]$Method = "GET", [hashtable]$Body = $null, [double]$TimeoutSeconds = 600) {
  $params = @{
    Uri = $Url
    Method = $Method
    TimeoutSec = $TimeoutSeconds
    ContentType = "application/json"
    Headers = @{ "Accept" = "application/json" }
    UseBasicParsing = $true
  }
  if ($null -ne $Body) {
    # PS 5.1: avoid deep nesting issues; depth 20 is plenty for this payload.
    $params.Body = ($Body | ConvertTo-Json -Depth 20 -Compress)
  }
  return Invoke-RestMethod @params
}

function DiscoverModels([string]$HostBase, [double]$TimeoutSeconds) {
  $tags = HttpJson -Url "$HostBase/api/tags" -TimeoutSeconds $TimeoutSeconds
  $names = @()
  $models = TryGetProp $tags "models"
  if ($null -ne $models) {
    foreach ($m in $models) {
      $n = TryGetProp $m "name"
      if ($null -ne $n -and "$n".Trim().Length -gt 0) {
        $names += "$n".Trim()
      }
    }
  }
  $names = $names | Sort-Object -Unique
  if ($names.Count -eq 0) {
    throw "No models found from /api/tags. Is Ollama running and has models been pulled?"
  }
  return ,$names
}

function GenerateOnce(
  [string]$HostBase,
  [string]$Model,
  [string]$PromptText,
  [double]$TimeoutSeconds,
  [hashtable]$Options,
  [string]$KeepAliveValue
) {
  $body = @{
    model = $Model
    prompt = $PromptText
    stream = $false
  }
  if ($Options.Count -gt 0) { $body.options = $Options }
  if ($null -ne $KeepAliveValue) { $body.keep_alive = $KeepAliveValue }

  $wallMs = $null
  try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = HttpJson -Url "$HostBase/api/generate" -Method "POST" -Body $body -TimeoutSeconds $TimeoutSeconds
    $sw.Stop()
    $wallMs = $sw.Elapsed.TotalSeconds
  } catch {
    if ($null -eq $wallMs) { $wallMs = 0.0 }
    return [pscustomobject]@{
      model = $Model
      ok = $false
      error = "$($_.Exception.Message)"
      wall_s = $wallMs
      load_s = $null
      total_s = $null
      prompt_eval_s = $null
      eval_s = $null
      prompt_tokens = $null
      gen_tokens = $null
      prompt_tps = $null
      gen_tps = $null
    }
  }

  $apiError = TryGetProp $resp "error"
  if ($null -ne $apiError -and "$apiError".Trim().Length -gt 0) {
    return [pscustomobject]@{
      model = $Model
      ok = $false
      error = "$apiError"
      wall_s = $wallMs
      load_s = $null
      total_s = $null
      prompt_eval_s = $null
      eval_s = $null
      prompt_tokens = $null
      gen_tokens = $null
      prompt_tps = $null
      gen_tps = $null
    }
  }

  $loadS = NsToS (TryGetProp $resp "load_duration")
  $totalS = NsToS (TryGetProp $resp "total_duration")
  $promptEvalS = NsToS (TryGetProp $resp "prompt_eval_duration")
  $evalS = NsToS (TryGetProp $resp "eval_duration")
  $promptEvalCount = TryGetProp $resp "prompt_eval_count"
  $evalCount = TryGetProp $resp "eval_count"
  $promptTokens = if ($null -ne $promptEvalCount) { [int]$promptEvalCount } else { $null }
  $genTokens = if ($null -ne $evalCount) { [int]$evalCount } else { $null }

  $promptTps = SafeDiv ([Nullable[double]]$promptTokens) ([Nullable[double]]$promptEvalS)
  $genTps = SafeDiv ([Nullable[double]]$genTokens) ([Nullable[double]]$evalS)

  return [pscustomobject]@{
    model = $Model
    ok = $true
    error = $null
    wall_s = $wallMs
    load_s = $loadS
    total_s = $totalS
    prompt_eval_s = $promptEvalS
    eval_s = $evalS
    prompt_tokens = $promptTokens
    gen_tokens = $genTokens
    prompt_tps = $promptTps
    gen_tps = $genTps
  }
}

function Aggregate([object[]]$Results) {
  $ok = @($Results | Where-Object { $_.ok })
  $errs = @($Results | Where-Object { -not $_.ok })

  $genTps = @($ok | Where-Object { $null -ne $_.gen_tps } | ForEach-Object { [double]$_.gen_tps })
  $promptTps = @($ok | Where-Object { $null -ne $_.prompt_tps } | ForEach-Object { [double]$_.prompt_tps })
  $wallS = @($ok | Where-Object { $null -ne $_.wall_s } | ForEach-Object { [double]$_.wall_s })
  $totalS = @($ok | Where-Object { $null -ne $_.total_s } | ForEach-Object { [double]$_.total_s })

  return [pscustomobject]@{
    runs = $Results.Count
    ok_runs = $ok.Count
    err_runs = $errs.Count
    gen_tps_mean = (Mean $genTps)
    gen_tps_stdev = (SampleStdev $genTps)
    gen_tps_p50 = (Percentile $genTps 50)
    gen_tps_p90 = (Percentile $genTps 90)
    prompt_tps_mean = (Mean $promptTps)
    total_s_mean = (Mean $totalS)
    wall_s_mean = (Mean $wallS)
    errors = @($errs | Where-Object { $null -ne $_.error -and "$($_.error)".Length -gt 0 } | ForEach-Object { $_.error })
  }
}

function RenderReport(
  [string]$StartedAt,
  [string]$HostBase,
  [string[]]$ModelList,
  [string]$PromptDesc,
  [int]$RunsCount,
  [int]$WarmupCount,
  [double]$TimeoutSeconds,
  [hashtable]$Options,
  [string]$KeepAliveValue,
  [hashtable]$AllResults
) {
  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# Ollama Benchmark Report")
  $lines.Add("")
  $lines.Add("- Started: ``$StartedAt``")
  $lines.Add("- Host: ``$HostBase``")
  $lines.Add("- PowerShell: ``$($PSVersionTable.PSVersion)``")
  $lines.Add("- Platform: ``$([System.Environment]::OSVersion.VersionString)``")
  $lines.Add("- Models: ``$([string]::Join(', ', $ModelList))``")
  $lines.Add("- Runs per model: ``$RunsCount`` (warmup: ``$WarmupCount``)")
  $lines.Add("- Timeout (s): ``$TimeoutSeconds``")
  if ($null -ne $KeepAliveValue) { $lines.Add("- keep_alive: ``$KeepAliveValue``") }
  if ($Options.Count -gt 0) { $lines.Add("- Options: ``$((ConvertTo-Json $Options -Compress))``") }
  $lines.Add("- Prompt: ``$PromptDesc``")
  $lines.Add("")

  $lines.Add("## Summary")
  $lines.Add("")
  $lines.Add("| Model | OK/Total | Gen tok/s (mean) | Gen tok/s (p50) | Gen tok/s (p90) | Gen tok/s (stdev) | Prompt tok/s (mean) | Total s (mean) | Wall s (mean) |")
  $lines.Add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
  foreach ($m in $ModelList) {
    $agg = Aggregate $AllResults[$m]
    $lines.Add((
      "| {0} | {1}/{2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |" -f
        (MdEscape $m),
        $agg.ok_runs, $agg.runs,
        (FmtFloat $agg.gen_tps_mean 2),
        (FmtFloat $agg.gen_tps_p50 2),
        (FmtFloat $agg.gen_tps_p90 2),
        (FmtFloat $agg.gen_tps_stdev 2),
        (FmtFloat $agg.prompt_tps_mean 2),
        (FmtFloat $agg.total_s_mean 2),
        (FmtFloat $agg.wall_s_mean 2)
    ))
  }
  $lines.Add("")

  $lines.Add("## Details")
  $lines.Add("")
  foreach ($m in $ModelList) {
    $lines.Add("### $m")
    $lines.Add("")
    $lines.Add("| Run | OK | Gen tok/s | Prompt tok/s | Gen toks | Prompt toks | Eval s | Prompt eval s | Load s | Total s | Wall s | Error |")
    $lines.Add("|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    $i = 0
    foreach ($r in $AllResults[$m]) {
      $i++
      $lines.Add((
        "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} |" -f
          $i,
          ($(if ($r.ok) { "Y" } else { "N" })),
          (FmtFloat $r.gen_tps 2),
          (FmtFloat $r.prompt_tps 2),
          (FmtInt $r.gen_tokens),
          (FmtInt $r.prompt_tokens),
          (FmtFloat $r.eval_s 2),
          (FmtFloat $r.prompt_eval_s 2),
          (FmtFloat $r.load_s 2),
          (FmtFloat $r.total_s 2),
          (FmtFloat $r.wall_s 2),
          (MdEscape ($(if ($null -ne $r.error) { "$($r.error)" } else { "" })))
      ))
    }

    $agg = Aggregate $AllResults[$m]
    if ($agg.errors.Count -gt 0) {
      $lines.Add("")
      $lines.Add("Errors:")
      foreach ($e in $agg.errors) { $lines.Add("- ``$e``") }
    }
    $lines.Add("")
  }

  return ($lines -join "`n")
}

$HostUrl = $HostUrl.TrimEnd("/")
$startedAt = IsoNowLocal

if (($Prompt -ne "") -and ($PromptFile -ne "")) {
  throw "Provide only one of -Prompt or -PromptFile"
}

$promptText = ""
$promptDesc = ""
if ($PromptFile -ne "") {
  $promptText = Get-Content -Raw -Encoding UTF8 $PromptFile
  $promptDesc = "file:$PromptFile"
} elseif ($Prompt -ne "") {
  $promptText = $Prompt
  $promptDesc = "inline"
} else {
  $promptText = @"
You are a benchmarking assistant.
Task: produce a long, deterministic output.
Output: write the integers from 1 to 2000 separated by a single space.
Do not add any other words.
"@
  $promptDesc = "default: integers 1..2000"
}

if ($Models.Count -eq 0) {
  $Models = DiscoverModels -HostBase $HostUrl -TimeoutSeconds $TimeoutS
}

$Models = @(
  $Models |
    ForEach-Object { "$_".Split(",") } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" } |
    Select-Object -Unique
)
if ($Models.Count -eq 0) {
  throw "No models provided / discovered."
}

$options = @{}
$options.num_predict = $NumPredict
$options.temperature = $Temperature
if ($null -ne $Seed) { $options.seed = [int]$Seed }
if ($null -ne $TopP) { $options.top_p = [double]$TopP }
if ($null -ne $NumCtx) { $options.num_ctx = [int]$NumCtx }
if ($Stop -ne "") {
  $stops = @($Stop.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
  $options.stop = $stops
}

$keepAliveValue = $null
if ($KeepAlive -ne "") {
  $keepAliveValue = $KeepAlive.Trim()
  if ($keepAliveValue -eq "0") { $keepAliveValue = "0s" }
}

if ($Out -eq "") {
  $ts = (Get-Date).ToString("yyyyMMdd-HHmmss")
  $Out = Join-Path "reports" "ollama-bench-$ts.md"
}

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$allResults = @{}
foreach ($m in $Models) { $allResults[$m] = @() }

if ($Warmup -gt 0) {
  foreach ($m in $Models) {
    for ($i = 0; $i -lt $Warmup; $i++) {
      [void](GenerateOnce -HostBase $HostUrl -Model $m -PromptText $promptText -TimeoutSeconds $TimeoutS -Options $options -KeepAliveValue $keepAliveValue)
    }
  }
}

foreach ($m in $Models) {
  for ($i = 0; $i -lt $Runs; $i++) {
    $r = GenerateOnce -HostBase $HostUrl -Model $m -PromptText $promptText -TimeoutSeconds $TimeoutS -Options $options -KeepAliveValue $keepAliveValue
    $allResults[$m] += $r
  }
}

$report = RenderReport -StartedAt $startedAt -HostBase $HostUrl -ModelList $Models -PromptDesc $promptDesc -RunsCount $Runs -WarmupCount $Warmup -TimeoutSeconds $TimeoutS -Options $options -KeepAliveValue $keepAliveValue -AllResults $allResults
Set-Content -Path $Out -Value ($report + "`n") -Encoding UTF8

Write-Output $Out
