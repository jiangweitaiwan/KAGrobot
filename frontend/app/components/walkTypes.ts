// frontend/app/components/walkTypes.ts

export type WalkPathNode = { eid: string; label: string; isRc: boolean };

export type FrontierPreviewItem = { eid: string; label: string; is_rc: boolean };

// A step's frontier that has since been resolved (one candidate chosen, rest kept as history)
export type FrontierHistoryStep = {
  fromEid: string;
  candidates: FrontierPreviewItem[]; // excludes the one that was chosen
};

export type WalkPathStatus = "pending" | "active" | "dead_end";

export type WalkPathState = {
  pathId: number;
  visited: WalkPathNode[];              // ordered, start..current, replaces visitedLabels
  currentEid: string;
  currentLabel: string;
  status: WalkPathStatus;
  deadEndReason?: string;
  frontierPreview: FrontierPreviewItem[];      // this step's live, unresolved candidates
  frontierHistory: FrontierHistoryStep[];       // every resolved step's unchosen branches
};

export type WalkQAEntry = {
  pathId: number;
  question: string;
  answer?: string; // undefined while waiting for the user
};

export type WalkOverallStatus = "connecting" | "walking" | "awaiting_answer" | "complete" | "error";