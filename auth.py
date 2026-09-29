"""
auth.py — Bearer-token authentication and authorization.

The agent does not own user accounts: it trusts JWTs issued by the
operator's identity provider (the same login the customer app uses).
Expected claims:

    sub   — the client's id (as a string, per the JWT spec) for client tokens
    role  — "client" (default) or "staff"
    exp   — required; tokens without an expiry are rejected

Client tokens can only act on their own client id. Staff tokens can read
any client's risk score but cannot chat or confirm plans on a client's
behalf — those endpoints are client-only.
"""
import time
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config

ROLE_CLIENT = "client"
ROLE_STAFF = "staff"

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str
    client_id: Optional[int]  # set only for client tokens


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _verification_key(algorithm: str) -> str:
    return config.JWT_SECRET if algorithm.startswith("HS") else config.JWT_PUBLIC_KEY


def decode_token(token: str) -> Principal:
    """Verifies signature, expiry and (if configured) issuer/audience."""
    try:
        header_alg = jwt.get_unverified_header(token).get("alg", "")
        if header_alg not in config.JWT_ALGORITHMS:
            raise _unauthorized("Invalid token.")
        claims = jwt.decode(
            token,
            _verification_key(header_alg),
            algorithms=config.JWT_ALGORITHMS,
            issuer=config.JWT_ISSUER or None,
            audience=config.JWT_AUDIENCE or None,
            options={"require": ["exp", "sub"], "verify_aud": bool(config.JWT_AUDIENCE)},
            leeway=30,
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token expired.")
    except jwt.PyJWTError:
        raise _unauthorized("Invalid token.")

    role = claims.get("role", ROLE_CLIENT)
    subject = str(claims["sub"])
    if role == ROLE_STAFF:
        return Principal(subject=subject, role=ROLE_STAFF, client_id=None)
    if role != ROLE_CLIENT:
        raise _unauthorized("Invalid token.")
    try:
        client_id = int(subject)
    except ValueError:
        raise _unauthorized("Invalid token.")
    return Principal(subject=subject, role=ROLE_CLIENT, client_id=client_id)


def authenticate(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Missing bearer token.")
    return decode_token(credentials.credentials)


def require_client(principal: Principal = Depends(authenticate)) -> Principal:
    if principal.role != ROLE_CLIENT:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Client token required.")
    return principal


def require_staff(principal: Principal = Depends(authenticate)) -> Principal:
    if principal.role != ROLE_STAFF:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Staff token required.")
    return principal


def ensure_same_client(principal: Principal, client_id: Optional[int]) -> int:
    """Returns the caller's client id; rejects a request naming someone else.

    `client_id` in request bodies is kept for backward compatibility but is
    never trusted — the token decides who the caller is.
    """
    if client_id is not None and client_id != principal.client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed for this client.")
    return principal.client_id


def issue_token(subject: str, role: str = ROLE_CLIENT, ttl_seconds: int = 3600) -> str:
    """Signs an HS256 token with JWT_SECRET. For local development and tests
    only — in production tokens come from the identity provider."""
    if not config.JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set.")
    now = int(time.time())
    claims = {"sub": str(subject), "role": role, "iat": now, "exp": now + ttl_seconds}
    if config.JWT_ISSUER:
        claims["iss"] = config.JWT_ISSUER
    if config.JWT_AUDIENCE:
        claims["aud"] = config.JWT_AUDIENCE
    return jwt.encode(claims, config.JWT_SECRET, algorithm="HS256")
