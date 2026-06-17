# =============================================================================
# Adjacency pilot — live test driver.
#
# Prereqs:
#   - skb-redis container running on :6379
#   - skillbridge-pg container running on :5433 with current SSM jobs loaded
#   - API running:  uvicorn skillbridge.app:app --reload --port 8000
#   - .env has SESSION_STORE=redis + ADJACENCY_ACTIVATION_ENABLED=true
#
# Usage:
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow explicit
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow soft_offer_accept
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow soft_offer_decline
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow ordinal_followup
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow scope_digression
#   pwsh -File scripts\test_adjacency_live.ps1 -Flow all
#
# Each flow runs in a fresh session. Watch the API stderr for the
# `adjacent_recommendations turn=...` telemetry line on adjacency turns.
# =============================================================================
param(
    [ValidateSet('explicit', 'soft_offer_accept', 'soft_offer_decline',
                 'ordinal_followup', 'scope_digression', 'all')]
    [string]$Flow = 'all',

    [string]$BaseUrl = 'http://localhost:8000',

    # The credential-capped flows need a target role that maps to a job
    # in the DB where the lead match is band_capped_by_credential AND
    # the user can plausibly have the non-credential skills.
    # Defaults assume an automotive technician role (310S license cap);
    # override if your DB has a better-suited credential-capped job.
    [string]$CredentialCappedTargetRole = 'automotive technician',
    [string]$CredentialCappedSkillsText = (
        'engine diagnostics, brake service, ' +
        'electrical troubleshooting, customer service'
    ),

    # For flows that need a clean band-good or near-miss to start.
    [string]$AdjacencyOnlyTargetRole = 'warehouse worker',
    [string]$AdjacencyOnlySkillsText = (
        'forklift operation, shipping and receiving, ' +
        'inventory management, welding'
    )
)

$ErrorActionPreference = 'Stop'

# -----------------------------------------------------------------------------
# Chat helper. Threads session_id across turns. Prints per-turn diff so the
# transcript shows what fired without dumping raw JSON.
# -----------------------------------------------------------------------------
function Send-Chat {
    param(
        [string]$Message,
        [string]$SessionId
    )
    $body = @{ message = $Message }
    if ($SessionId) { $body['session_id'] = $SessionId }
    $json = $body | ConvertTo-Json -Compress

    $resp = Invoke-RestMethod -Method Post `
        -Uri "$BaseUrl/v1/chat/messages" `
        -ContentType 'application/json' `
        -Body $json

    $data = $resp.data
    return $data
}

function Show-Turn {
    param($Label, $UserMsg, $Data)
    $move = $Data.final_move
    $reply = $Data.reply
    if ($reply.Length -gt 280) {
        $replyShown = $reply.Substring(0, 280) + '...'
    } else {
        $replyShown = $reply
    }
    Write-Host ""
    Write-Host "--- $Label ---" -ForegroundColor Cyan
    Write-Host "USER: $UserMsg" -ForegroundColor Gray
    Write-Host "MOVE: $move" -ForegroundColor Yellow
    Write-Host "BOT : $replyShown"
}

# -----------------------------------------------------------------------------
# Shared setup paths
# -----------------------------------------------------------------------------
function Start-CredentialCappedSession {
    Write-Host ""
    Write-Host "==> Bootstrapping credential-capped session (target='$CredentialCappedTargetRole')" -ForegroundColor Green
    $d = Send-Chat -Message "I'm looking for work as a $CredentialCappedTargetRole"
    $sid = $d.session_id
    Show-Turn 'turn 1 (target)' "I'm looking for work as a $CredentialCappedTargetRole" $d

    $d = Send-Chat -Message "My skills: $CredentialCappedSkillsText" -SessionId $sid
    Show-Turn 'turn 2 (skills)' "My skills: $CredentialCappedSkillsText" $d

    # Some intake states need one more nudge to surface matches.
    if ($d.final_move -notin @('present_matches', 'present_near_miss', 'present_no_match')) {
        $d = Send-Chat -Message "I have 3 years of experience in this work" -SessionId $sid
        Show-Turn 'turn 3 (experience)' 'I have 3 years of experience in this work' $d
    }
    return $sid
}

function Start-AdjacencyOnlySession {
    Write-Host ""
    Write-Host "==> Bootstrapping band-good session (target='$AdjacencyOnlyTargetRole')" -ForegroundColor Green
    $d = Send-Chat -Message "I'm looking for work as a $AdjacencyOnlyTargetRole"
    $sid = $d.session_id
    Show-Turn 'turn 1 (target)' "I'm looking for work as a $AdjacencyOnlyTargetRole" $d

    $d = Send-Chat -Message "My skills: $AdjacencyOnlySkillsText" -SessionId $sid
    Show-Turn 'turn 2 (skills)' "My skills: $AdjacencyOnlySkillsText" $d

    if ($d.final_move -notin @('present_matches', 'present_near_miss', 'present_no_match')) {
        $d = Send-Chat -Message "I have 3 years of experience in this work" -SessionId $sid
        Show-Turn 'turn 3 (experience)' 'I have 3 years of experience in this work' $d
    }
    return $sid, $d
}

# -----------------------------------------------------------------------------
# Flow 1: explicit request. User asks for related roles WITHOUT a soft offer.
# Expected: final_move=recommend_adjacent_roles. trigger=user_explicit in
# the API telemetry log.
# -----------------------------------------------------------------------------
function Test-ExplicitRequest {
    Write-Host ""
    Write-Host "##############################################" -ForegroundColor Magenta
    Write-Host "# FLOW 1: explicit recommend request"          -ForegroundColor Magenta
    Write-Host "##############################################" -ForegroundColor Magenta
    $sid, $d = Start-AdjacencyOnlySession
    $d = Send-Chat -Message 'what other roles could I look at?' -SessionId $sid
    Show-Turn 'EXPLICIT REQUEST' 'what other roles could I look at?' $d
    Write-Host ""
    Write-Host "EXPECT: final_move=recommend_adjacent_roles" -ForegroundColor DarkCyan
    Write-Host "EXPECT: API log line 'adjacent_recommendations ... trigger=user_explicit'" -ForegroundColor DarkCyan
}

# -----------------------------------------------------------------------------
# Flow 2: soft-offer accept. User's stated target gets credential-capped,
# soft offer fires, user says yes.
# Expected: turn N reply ends with the soft-offer line; turn N+1
# final_move=recommend_adjacent_roles + trigger=soft_offer_accepted.
# -----------------------------------------------------------------------------
function Test-SoftOfferAccept {
    Write-Host ""
    Write-Host "##############################################" -ForegroundColor Magenta
    Write-Host "# FLOW 2: soft offer -> accept"                -ForegroundColor Magenta
    Write-Host "##############################################" -ForegroundColor Magenta
    $sid = Start-CredentialCappedSession
    Write-Host ""
    Write-Host "PRE-CHECK: prior bot reply should end with 'just say *what other roles?*'" -ForegroundColor DarkCyan
    Write-Host "If not, the lead match was not credential-only capped; pick a different target role." -ForegroundColor DarkYellow

    $d = Send-Chat -Message 'yes please' -SessionId $sid
    Show-Turn 'ACCEPT' 'yes please' $d
    Write-Host ""
    Write-Host "EXPECT: final_move=recommend_adjacent_roles" -ForegroundColor DarkCyan
    Write-Host "EXPECT: API log line 'adjacent_recommendations ... trigger=soft_offer_accepted'" -ForegroundColor DarkCyan
}

# -----------------------------------------------------------------------------
# Flow 3: soft-offer decline. Same setup as flow 2, but user declines.
# Expected: turn N+1 final_move != recommend_adjacent_roles; reply does NOT
# end with the soft-offer line (reoffer suppression).
# -----------------------------------------------------------------------------
function Test-SoftOfferDecline {
    Write-Host ""
    Write-Host "##############################################" -ForegroundColor Magenta
    Write-Host "# FLOW 3: soft offer -> decline (NO reoffer)"  -ForegroundColor Magenta
    Write-Host "##############################################" -ForegroundColor Magenta
    $sid = Start-CredentialCappedSession

    $d = Send-Chat -Message 'no thanks' -SessionId $sid
    Show-Turn 'DECLINE' 'no thanks' $d
    Write-Host ""
    if ($d.reply -match 'what other roles\?') {
        Write-Host "FAIL: reoffer line present after decline" -ForegroundColor Red
    } else {
        Write-Host "OK  : no reoffer line in reply" -ForegroundColor Green
    }
    Write-Host "EXPECT: final_move is NOT recommend_adjacent_roles" -ForegroundColor DarkCyan
    Write-Host "EXPECT: reply does NOT end with 'just say *what other roles?*'" -ForegroundColor DarkCyan
}

# -----------------------------------------------------------------------------
# Flow 4: ordinal follow-up. Triggers recommend_adjacent_roles, then asks
# 'tell me about the second one'.
# Expected: final_move=describe_adjacent_role, with the second role narrated.
# -----------------------------------------------------------------------------
function Test-OrdinalFollowup {
    Write-Host ""
    Write-Host "##############################################" -ForegroundColor Magenta
    Write-Host "# FLOW 4: ordinal follow-up"                   -ForegroundColor Magenta
    Write-Host "##############################################" -ForegroundColor Magenta
    $sid, $d = Start-AdjacencyOnlySession
    $d = Send-Chat -Message 'what other roles could I look at?' -SessionId $sid
    Show-Turn 'RECOMMEND' 'what other roles could I look at?' $d

    if ($d.final_move -ne 'recommend_adjacent_roles') {
        Write-Host "SKIP: prior turn did not yield recommend_adjacent_roles (got $($d.final_move))" -ForegroundColor Red
        return
    }

    $d = Send-Chat -Message 'tell me about the second one' -SessionId $sid
    Show-Turn 'ORDINAL' 'tell me about the second one' $d
    Write-Host ""
    Write-Host "EXPECT: final_move=describe_adjacent_role" -ForegroundColor DarkCyan
    Write-Host "EXPECT: reply names ONE specific role (not a list)" -ForegroundColor DarkCyan
}

# -----------------------------------------------------------------------------
# Flow 5: scope digression + recovery. Triggers recommend, digresses, returns.
# Expected: digression turn -> redirect_scope; recovery turn ->
# describe_adjacent_role (snapshot survived because TTL was shifted).
# -----------------------------------------------------------------------------
function Test-ScopeDigression {
    Write-Host ""
    Write-Host "##############################################" -ForegroundColor Magenta
    Write-Host "# FLOW 5: scope digression + recovery"         -ForegroundColor Magenta
    Write-Host "##############################################" -ForegroundColor Magenta
    $sid, $d = Start-AdjacencyOnlySession
    $d = Send-Chat -Message 'what other roles could I look at?' -SessionId $sid
    Show-Turn 'RECOMMEND' 'what other roles could I look at?' $d

    if ($d.final_move -ne 'recommend_adjacent_roles') {
        Write-Host "SKIP: prior turn did not yield recommend_adjacent_roles (got $($d.final_move))" -ForegroundColor Red
        return
    }

    $d = Send-Chat -Message 'how do I get my PR card?' -SessionId $sid
    Show-Turn 'DIGRESSION' 'how do I get my PR card?' $d
    Write-Host "EXPECT: final_move=redirect_scope" -ForegroundColor DarkCyan

    $d = Send-Chat -Message 'tell me about the second one' -SessionId $sid
    Show-Turn 'RECOVERY' 'tell me about the second one' $d
    Write-Host ""
    Write-Host "EXPECT: final_move=describe_adjacent_role (snapshot survived 1-turn digression)" -ForegroundColor DarkCyan
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
switch ($Flow) {
    'explicit'           { Test-ExplicitRequest }
    'soft_offer_accept'  { Test-SoftOfferAccept }
    'soft_offer_decline' { Test-SoftOfferDecline }
    'ordinal_followup'   { Test-OrdinalFollowup }
    'scope_digression'   { Test-ScopeDigression }
    'all' {
        Test-ExplicitRequest
        Test-SoftOfferAccept
        Test-SoftOfferDecline
        Test-OrdinalFollowup
        Test-ScopeDigression
    }
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Done. Check the API stderr for telemetry lines:" -ForegroundColor Green
Write-Host "  adjacent_recommendations turn=... candidates_returned=..." -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
