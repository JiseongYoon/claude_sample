"""Storage connector capability.

Permissioned remote file access behind a pluggable connector seam. ships the seam
itself — the `RemoteTransport` Protocol (raw remote ops, implemented by concrete connectors in
later steps), the `GuardedConnector` policy wrapper (containment + read-only + size cap), the
typed `StorageError` hierarchy, and the connector/credential config model — all network-free
and testable with an injected fake transport.
"""
