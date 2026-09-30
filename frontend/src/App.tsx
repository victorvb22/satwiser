import { QueryClient, QueryClientProvider, useQueryClient } from "@tanstack/react-query";
import { lazy, Suspense, useEffect, useState } from "react";
import { createBrowserRouter, RouterProvider, type RouteObject } from "react-router-dom";

import { queries } from "./api/client";
import { Layout, Loading } from "./components/Layout";
import Mission from "./pages/Mission";

const Lab = lazy(() => import("./pages/Lab"));
const Method = lazy(() => import("./pages/Method"));

export const queryClient = new QueryClient({
  defaultOptions: {
    // The hosted API sleeps when idle: retry patiently instead of failing fast.
    queries: { retry: 4, retryDelay: (n) => Math.min(2000 * 2 ** n, 15000),
               refetchOnWindowFocus: false },
  },
});

export const routes: RouteObject[] = [
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Mission /> },
      { path: "/labo", element: <Suspense fallback={<Loading what="du labo" />}><Lab /></Suspense> },
      { path: "/labo/:eventId",
        element: <Suspense fallback={<Loading what="du labo" />}><Lab /></Suspense> },
      { path: "/methode",
        element: <Suspense fallback={<Loading what="de la page" />}><Method /></Suspense> },
    ],
  },
];

function Prefetch() {
  const client = useQueryClient();
  useEffect(() => {
    void client.prefetchQuery(queries.satellites());
    void client.prefetchQuery(queries.robustness());
  }, [client]);
  return null;
}

type AppRouter = ReturnType<typeof createBrowserRouter>;

export default function App({ client = queryClient, router }: { client?: QueryClient;
                                                               router?: AppRouter }) {
  const [browserRouter] = useState(() => router ?? createBrowserRouter(routes));
  return (
    <QueryClientProvider client={client}>
      <Prefetch />
      <RouterProvider router={browserRouter} />
    </QueryClientProvider>
  );
}
