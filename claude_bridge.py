#!/usr/bin/env python3
"""
Python bridge script for Claude Code SDK with OAuth authentication
"""
import asyncio
import json
import sys
import argparse
from claude_code_sdk import query_with_oauth, ClaudeCodeOptions


async def claude_query(prompt: str, options: dict = None):
    """Execute Claude Code query with OAuth authentication"""
    try:
        # Parse options
        claude_options = None
        if options:
            claude_options = ClaudeCodeOptions(
                system_prompt=options.get('customSystemPrompt'),
                max_turns=options.get('maxTurns', 10),
                allowed_tools=options.get('allowedTools', []),
                cwd=options.get('cwd', '.'),
                resume=options.get('resume')
            )
        
        messages = []
        
        # Use OAuth authentication instead of API key
        async for message in query_with_oauth(prompt=prompt, options=claude_options):
            # Convert message to JSON serializable format
            message_dict = {
                'type': getattr(message, 'type', 'unknown'),
                'content': getattr(message, 'content', ''),
                'subtype': getattr(message, 'subtype', None),
                'session_id': getattr(message, 'session_id', None)
            }
            
            # Add any other attributes that might be present
            for attr in dir(message):
                if not attr.startswith('_') and attr not in ['type', 'content', 'subtype', 'session_id']:
                    try:
                        value = getattr(message, attr)
                        if not callable(value):
                            message_dict[attr] = value
                    except:
                        pass
            
            messages.append(message_dict)
            
            # Stream output for real-time feedback
            print(json.dumps(message_dict), flush=True)
        
        return messages
        
    except Exception as e:
        error_message = {
            'type': 'error',
            'content': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_message), flush=True)
        return [error_message]


async def main():
    parser = argparse.ArgumentParser(description='Claude Code OAuth Bridge')
    parser.add_argument('--prompt', required=True, help='Prompt to send to Claude')
    parser.add_argument('--options', help='JSON string with options')
    
    args = parser.parse_args()
    
    # Parse options if provided
    options = None
    if args.options:
        try:
            options = json.loads(args.options)
        except json.JSONDecodeError as e:
            error = {
                'type': 'error',
                'content': f'Invalid JSON in options: {e}',
                'error_type': 'JSONDecodeError'
            }
            print(json.dumps(error))
            sys.exit(1)
    
    # Execute query
    await claude_query(args.prompt, options)


if __name__ == '__main__':
    asyncio.run(main())