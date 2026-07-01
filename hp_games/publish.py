from identdynamics import default_platform_tag
from identdynamics import Client, build_protected_bundle
import uuid
print(default_platform_tag())           # e.g. "macos-arm64-cp314"
print(hex(uuid.getnode()))              # MAC address


#linux=True if x platform build

out = build_protected_bundle(
    "pub_src",                        # folder containing anom_ca_pub.py
    "dist/brahma_ca_03252_anomaly",    # output dir
    model_name="brahma_ca_03252_anomaly",
    model_version="1.0.0",
    expire="2027-01-01",
    # bind_mac=["0x184a53232c2c"],         # from step 1
    harden=False,
)

client = Client("https://goident.ai", token="1e3464d23383b10c52515ab7978e4d7ff8fca3e982ba1202c5be6677ea290c48")

client.publish_protected(
    "brahma_ca_03252_anomaly",
    out,
    platform="darwin-arm64-cp314",  # from step 1
    grant=["rdtandon"],
)


result = client.grant_protected("brahma_ca_03252_anomaly", ["rdtandon"])
print("grant:", result)

print("***")
import json
listed = client.list_protected_published()
print(json.dumps(listed if isinstance(listed, (list, dict)) else list(listed), indent=2))


