#!/usr/bin/env python3
"""CLI to manage MCP desktop environment addresses.

Usage:
    # List all allocated addresses
    python -m recipe.gui_agent.scripts.mcp_manage list --env huaiyizhao

    # Release all addresses (e.g. after a crashed training run)
    python -m recipe.gui_agent.scripts.mcp_manage release --env huaiyizhao

    # Dry-run: see what would be released without actually releasing
    python -m recipe.gui_agent.scripts.mcp_manage release --env huaiyizhao --dry-run
"""

import argparse
import asyncio
import sys

sys.path.insert(0, ".")
from cua.galileo.tools.utils.mcp_address import ComputerUseMCPAddressClient


async def cmd_list(client: ComputerUseMCPAddressClient):
    addresses = await client.list_all_addresses()
    if not addresses:
        print("No allocated addresses.")
        return
    print(f"Found {len(addresses)} allocated address(es):\n")
    for addr in addresses:
        print(f"  {addr['address']}  ({addr.get('service_name', 'N/A')}, {addr.get('phase', '?')})")


async def cmd_release(client: ComputerUseMCPAddressClient, dry_run: bool = False):
    result = await client.unlock_all_addresses(dry_run=dry_run)
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Total: {result['total']}, Unlocked: {result['unlocked']}, Failed: {result['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Manage MCP desktop environment addresses")
    parser.add_argument("command", choices=["list", "release"], help="Command to run")
    parser.add_argument("--base-url", default="http://ns008-cu-manager-svc", help="MCP manager base URL")
    parser.add_argument("--env", required=True, help="Environment identifier (e.g. huaiyizhao)")
    parser.add_argument("--namespace", default="Development", help="Namespace")
    parser.add_argument("--dry-run", action="store_true", help="List only, don't release")
    args = parser.parse_args()

    client = ComputerUseMCPAddressClient(
        base_url=args.base_url,
        env=args.env,
        namespace=args.namespace,
    )

    if args.command == "list":
        asyncio.run(cmd_list(client))
    elif args.command == "release":
        asyncio.run(cmd_release(client, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
