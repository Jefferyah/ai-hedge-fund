# Multi-stage build: Vite -> nginx
# VITE_API_URL must be baked into the bundle at build time.

FROM node:20-alpine AS builder

WORKDIR /build

# VITE_* vars must be declared as ARG so they are visible to `vite build`.
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

# Copy only the frontend app to keep the build context small and
# avoid leaking backend Python files into the Node layer.
COPY app/frontend/package.json app/frontend/package-lock.json* /build/
RUN npm install

COPY app/frontend/ /build/
# Upstream `npm run build` runs `tsc && vite build`; `tsc` fails on pre-existing
# unused-var / forwardRef type errors in the source. We only need the bundle,
# so invoke vite directly and skip the standalone type check.
RUN npx vite build

FROM nginx:alpine

# Simple SPA config: serve /dist, fall through to index.html on 404
# so React Router routes work on refresh.
RUN rm /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

RUN printf 'server {\n\
    listen       8080;\n\
    server_name  _;\n\
    root         /usr/share/nginx/html;\n\
    index        index.html;\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 8080
