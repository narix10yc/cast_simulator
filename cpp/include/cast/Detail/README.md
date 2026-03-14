# Internal header tier

Headers under `cast/Detail/` are not part of the intended stable public API.

They may still be transitively included by public headers during migration, but new implementation-only details should move here instead of `cast/Core/` or `cast/Internal/`.
