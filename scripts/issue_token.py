"""
scripts/issue_token.py — Mint a signed JWT for local development and testing.

In production, tokens are issued by the operator's identity provider; this
script only exists so you can call the API locally. It signs with JWT_SECRET
from your environment / .env.

    python scripts/issue_token.py --client-id 3
    python scripts/issue_token.py --staff
    curl -H "Authorization: Bearer $(python scripts/issue_token.py --client-id 3)" \
         http://localhost:8000/client/assistant/auto-session/3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import ROLE_CLIENT, ROLE_STAFF, issue_token  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--client-id", type=int, help="issue a client token for this client id")
    who.add_argument("--staff", metavar="NAME", nargs="?", const="staff", help="issue a staff token")
    parser.add_argument("--ttl-minutes", type=int, default=60)
    args = parser.parse_args()

    if args.client_id is not None:
        token = issue_token(str(args.client_id), ROLE_CLIENT, args.ttl_minutes * 60)
    else:
        token = issue_token(args.staff, ROLE_STAFF, args.ttl_minutes * 60)
    print(token)


if __name__ == "__main__":
    main()
