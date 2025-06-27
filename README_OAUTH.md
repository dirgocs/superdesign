# OAuth Integration Setup

This project has been modified to use OAuth authentication instead of API key authentication for Claude Code interactions.

## Setup

1. **Python Virtual Environment**: A virtual environment `claude-sdk-venv` has been created with the alternative Claude Code SDK installed.

2. **Bridge Script**: `claude_bridge.py` acts as a bridge between the TypeScript extension and the Python SDK that supports OAuth.

3. **Modified Service**: `claudeCodeService.ts` now uses subprocess calls to the Python bridge instead of the official SDK.

## OAuth Flow

When you use the extension, it will:
1. Launch the Python bridge script
2. The bridge script will initiate OAuth authentication 
3. A browser window will open for you to authenticate with your Claude Code subscription
4. Once authenticated, the extension will use your subscription access

## Key Changes

- Removed dependency on `@anthropic-ai/claude-code` 
- Added Python bridge script for OAuth authentication
- Modified TypeScript service to use subprocess communication
- Updated build process to exclude Claude SDK copying

## Benefits

- Uses subscription OAuth instead of API key
- Bypasses API rate limits
- Leverages Claude Code Max features programmatically

## Files Modified

- `package.json` - Removed official SDK dependency
- `src/services/claudeCodeService.ts` - Added Python bridge integration
- `esbuild.js` - Removed Claude SDK copying
- `claude_bridge.py` - New Python bridge script
- `claude-sdk-venv/` - Python virtual environment with alternative SDK

## First Time Setup

On first use, the extension will trigger OAuth authentication. Follow the browser prompts to authenticate with your Claude Code subscription.