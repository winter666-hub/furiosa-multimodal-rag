const CLIENT_IP_HEADER = "CF-Connecting-IP";

export function paperRagProxyHeaders(headers: Headers): Record<string, string> {
  const proxySecret = process.env.PAPER_RAG_PROXY_SECRET;
  const clientIp = headers.get(CLIENT_IP_HEADER)?.trim();
  if (!proxySecret || !clientIp || clientIp.length > 128) return {};
  return {
    "X-Paper-RAG-Client-IP": clientIp,
    "X-Paper-RAG-Proxy-Token": proxySecret,
  };
}
