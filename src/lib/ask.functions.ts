import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { askBackend, checkBackendHealth } from "./ask.server";

export const askPaperQuestion = createServerFn({ method: "POST" })
  .inputValidator((data) =>
    z.object({ question: z.string().trim().min(1).max(2000) }).parse(data),
  )
  .handler(async ({ data }) => askBackend(data.question));

// Also acts as a warm-up ping so a sleeping backend starts waking
// before the user sends their first question.
export const getBackendHealth = createServerFn({ method: "GET" }).handler(
  async () => checkBackendHealth(),
);
