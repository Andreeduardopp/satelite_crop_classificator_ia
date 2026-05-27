"""
List all available Sentinel Hub collections via the Catalog API.
"""

import requests
from sentinelhub import SHConfig

TOKEN_URL = "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
CATALOG_URL = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/collections"


def get_access_token(config: SHConfig) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": config.sh_client_id,
            "client_secret": config.sh_client_secret,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_collections(token: str) -> list[dict]:
    resp = requests.get(
        CATALOG_URL,
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json().get("collections", [])


if __name__ == "__main__":
    config = SHConfig()
    config.sh_client_id = '8103d553-6466-4a14-8861-a59f11dc7387'
    config.sh_client_secret = '4Ewmc2uMHghjubNUnWfVkhDlGztPo4o8'
    if not config.sh_client_id or not config.sh_client_secret:
        print("ERROR: sh_client_id / sh_client_secret not set in SHConfig.")
        print("Run once:")
        print('  config = SHConfig()')
        print('  config.sh_client_id = "YOUR_CLIENT_ID"')
        print('  config.sh_client_secret = "YOUR_SECRET"')
        print('  config.save()')
        raise SystemExit(1)

    token = get_access_token(config)
    collections = list_collections(token)

    print(f"\n{'='*60}")
    print(f" Found {len(collections)} collections")
    print(f"{'='*60}\n")

    for c in sorted(collections, key=lambda x: x.get("id", "")):
        cid = c.get("id", "?")
        title = c.get("title", "")
        desc = c.get("description", "")[:120]
        print(f"  {cid}")
        if title:
            print(f"    title: {title}")
        if desc:
            print(f"    desc:  {desc}")
        print()
