export function onRoute(route: string): boolean {
  const current = window.location.hash.replace(/^#\/?/, "").split("?")[0];
  return current === route || (!current && route === "study");
}
