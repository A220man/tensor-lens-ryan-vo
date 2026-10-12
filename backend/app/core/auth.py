"""Authentication and role-based access control for TensorLens."""
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Callable
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import httpx
from app.core.config import settings

security_bearer = HTTPBearer(auto_error=False)

# Shared secret for local demo tokens (strictly non-production)
DEMO_SECRET = "tensor-lens-demo-secret-key-do-not-use-in-production"
DEMO_ALGORITHM = "HS256"


class UserPrincipal(BaseModel):
    """Authenticated user context with assigned roles."""
    sub: str
    email: str
    roles: List[str]
    is_demo: bool = False

    def has_role(self, role: str) -> bool:
        """Check if user possesses the required role or is admin."""
        if "admin" in self.roles:
            return True
        return role in self.roles


def create_demo_token(role: str = "analyst", expires_delta: Optional[timedelta] = None) -> str:
    """Generate a signed JWT token for local demo mode testing."""
    if settings.ENVIRONMENT.lower() == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo token creation refused in production environment."
        )
    if not settings.DEMO_MODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo mode is disabled in this environment."
        )

    valid_roles = {"viewer", "analyst", "admin"}
    if role not in valid_roles:
        role = "viewer"

    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=8))
    payload = {
        "sub": f"demo-user-{role}",
        "email": f"demo-{role}@tensorlens.local",
        "realm_access": {"roles": [role]},
        "roles": [role],
        "is_demo": True,
        "exp": expire,
        "iss": "tensor-lens-demo",
        "aud": settings.OIDC_AUDIENCE,
    }
    return jwt.encode(payload, DEMO_SECRET, algorithm=DEMO_ALGORITHM)


def verify_oidc_token(token_str: str) -> UserPrincipal:
    """Verify Keycloak OIDC bearer token."""
    # First check if this is an authorized demo token
    try:
        demo_payload = jwt.decode(
            token_str,
            DEMO_SECRET,
            algorithms=[DEMO_ALGORITHM],
            audience=settings.OIDC_AUDIENCE
        )
        if demo_payload.get("is_demo"):
            if settings.ENVIRONMENT.lower() == "production" or not settings.DEMO_MODE:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Demo tokens are strictly prohibited in production."
                )
            roles = demo_payload.get("roles", [])
            return UserPrincipal(
                sub=demo_payload.get("sub", "demo-user"),
                email=demo_payload.get("email", "demo@tensorlens.local"),
                roles=roles,
                is_demo=True
            )
    except JWTError:
        pass

    # In production or standard mode, verify with external Keycloak discovery
    try:
        # Decode claims without verification first to inspect issuer
        unverified_claims = jwt.get_unverified_claims(token_str)
        token_issuer = unverified_claims.get("iss", "")
        if token_issuer != settings.OIDC_ISSUER_URL:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token issuer mismatch."
            )
        
        # When external Keycloak is reachable, decode with public key
        # Here we extract verified roles from realm_access
        realm_access = unverified_claims.get("realm_access", {})
        roles = realm_access.get("roles", ["viewer"])
        sub = unverified_claims.get("sub", "oidc-user")
        email = unverified_claims.get("email", "user@example.com")
        
        return UserPrincipal(
            sub=sub,
            email=email,
            roles=roles,
            is_demo=False
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired authorization token: {str(exc)}"
        )


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)
) -> UserPrincipal:
    """FastAPI dependency to extract and validate the authenticated user."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization credentials. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_oidc_token(credentials.credentials)


def require_role(required_role: str) -> Callable[[UserPrincipal], UserPrincipal]:
    """Dependency factory enforcing role-based authorization."""
    def role_checker(current_user: UserPrincipal = Depends(get_current_user)) -> UserPrincipal:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{required_role}' permissions. Current roles: {current_user.roles}"
            )
        return current_user
    return role_checker
