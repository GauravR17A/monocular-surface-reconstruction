# Security and research-use boundaries

This package is intended for local research use. The API binds to loopback by
default and has no authentication or multi-user result isolation. Do not expose
it as a public upload service without addressing the controls in the
[security review](docs/SECURITY_REVIEW.md).

For a suspected vulnerability, contact the repository owner privately through
their published GitHub contact channel. Share a minimal reproduction and affected
version; do not publish credentials, private uploads or an exploitable detail in
a public issue before it can be assessed. No response deadline or bounty is promised.

Only load trusted model artifacts. Keep `.env`, runtime results, credentials,
private data and hosting state outside Git. Source datasets, pretrained weights
and dependencies retain their separate terms.
