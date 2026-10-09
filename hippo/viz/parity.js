// Replays the exported event streams through the JS engine and checks it
// against the Python reference: per query, the same answer correctness, the
// same retrieved memory ids in the same order, and the same scores (to 1e-9
// relative), plus an identical final store.
//   node viz/parity.js
const fs = require("fs");
const path = require("path");
const E = require("./engine.js");

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const TOL = 1e-9;
let failed = false;

for (const [seed, s] of Object.entries(data.seeds)) {
  const agent = new E.Agent(E.defaultConfig());
  const trace = [], problems = [];
  let qi = 0;
  for (const [kind, text, slot, gold] of s.events) {
    if (kind === "o") agent.observe(text, slot);
    else {
      const res = agent.recall(text);
      const ok = E.readAnswer(slot, res) === gold;
      agent.feedback(ok ? 1 : -1);
      trace.push(ok ? 1 : 0);
      const ref = s.reference_retrieved[qi];
      const got = res.map((r) => [r.item.id, r.score]);
      if (problems.length < 3) {
        if (got.length !== ref.length || got.some((g, i) => g[0] !== ref[i][0])) {
          problems.push(`query #${qi}: retrieved ids ${JSON.stringify(got.map((g) => g[0]))} vs python ${JSON.stringify(ref.map((r) => r[0]))}`);
        } else {
          for (let i = 0; i < got.length; i++) {
            const d = Math.abs(got[i][1] - ref[i][1]);
            if (d > TOL * Math.max(1, Math.abs(ref[i][1]))) {
              problems.push(`query #${qi} rank ${i}: score ${got[i][1]} vs python ${ref[i][1]}`);
              break;
            }
          }
        }
      }
      qi++;
    }
    agent.maintain();
    agent.drain();
  }
  const ref = s.reference;
  let diffs = 0;
  for (let i = 0; i < ref.length; i++) if (ref[i] !== trace[i]) diffs++;
  const st = agent.store;
  const live = st.liveItems();
  const fin = {
    total: live.length,
    episodic: live.filter((i) => i.layer === "episodic").length,
    semantic: live.filter((i) => i.layer === "semantic").length,
    written: st.stats.written, pruned: st.stats.pruned, superseded: st.stats.superseded,
  };
  const finOk = JSON.stringify(fin) === JSON.stringify(s.reference_final);
  const acc = (a) => (a.reduce((x, y) => x + y, 0) / a.length).toFixed(3);
  const ok = !diffs && finOk && !problems.length;
  console.log(`seed ${seed}: ${ok ? "OK" : "MISMATCH"} | js acc ${acc(trace)} py acc ${acc(ref)} | ` +
              `answer mismatches ${diffs} | final state ${finOk ? "identical" : "DIFFERS"}`);
  for (const p of problems) console.log("   " + p);
  if (!finOk) console.log("   js", fin, "\n   py", s.reference_final);
  if (!ok) failed = true;
}
process.exit(failed ? 1 : 0);
