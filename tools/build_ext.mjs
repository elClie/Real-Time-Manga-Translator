// SPDX-License-Identifier: GPL-3.0-only
// Assembles the loadable extension in dist/extension: extension/ + pipeline/ + vendor libs + models.
// Big files (models, wasm) are hard links, so the ~0.6 GB costs no extra disk; sources are copied.
// usage: npm install && node tools/build_ext.mjs   then load dist/extension unpacked in chrome://extensions
import { cpSync, existsSync, linkSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, copyFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const OUT = join(ROOT, "dist", "extension");

const ORT_GLUE = "onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs";
const OPENCV = "@techstark/opencv-js/dist/opencv.js";
const VENDOR = ["onnxruntime-web/dist/ort.webgpu.min.mjs", "onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm", "clipper-lib/clipper.js"];
const MODELS = [
  "det16.onnx",
  "det16.bin",
  "PP-OCRv6_small_rec_top1.onnx",
  "ppocrv6_dict.txt",
  "mocr_encoder_fp16.onnx",
  "mocr_decoder_fp16.onnx",
  "mocr_vocab.json",
  "langid.bin",
  "langid.json",
  "unicode_tables.json",
  "panels.onnx",
];

function link(src, dst) {
  if (!existsSync(src)) throw new Error(`missing ${src}`);
  mkdirSync(dirname(dst), { recursive: true });
  try {
    linkSync(src, dst);
  } catch {
    copyFileSync(src, dst); // other volume
  }
}

// The MV3 extension CSP forbids compiling strings to code, and both Emscripten builds were made with
// embind's dynamic execution on. Each patched site generates a small wrapper from a fixed template;
// the replacements build the same wrapper from closures (what DYNAMIC_EXECUTION=0 would emit).
// Any change in the vendored code fails the build instead of silently shipping eval.

// ORT: _emval_create_invoker. The generated source only takes four shapes (call, method call, new,
// cast) over N pointer args.
const INVOKER = `function __panelProbeInvoker(src, env) {
  const body = src.slice(src.indexOf("{") + 1, src.lastIndexOf("}")).trim();
  const RET = "return emval_returnValue(toReturnWire, destructorsRef, ";
  const ret = body.startsWith(RET);
  const core = ret ? body.slice(RET.length, -1) : body;
  const n = (core.match(/argFromPtr\\d+/g) || []).length;
  const from = Array.from({ length: n }, (_, p) => env["argFromPtr" + p]);
  const argv = (a) => from.map((f, p) => f(a + 8 * p));
  let call;
  if (core.startsWith("toValue(handle)[getStringOrSymbol(methodName)](")) call = (h, m, a) => { const o = env.toValue(h); return o[env.getStringOrSymbol(m)](...argv(a)); };
  else if (core.startsWith("new (toValue(handle))(")) call = (h, m, a) => new (env.toValue(h))(...argv(a));
  else if (core.startsWith("toValue(handle)(")) call = (h, m, a) => env.toValue(h)(...argv(a));
  else if (core.startsWith("(") && n) call = (h, m, a) => argv(a)[n - 1];
  else throw new Error("unexpected emval invoker: " + body);
  return ret ? (h, m, d, a) => env.emval_returnValue(env.toReturnWire, d, call(h, m, a)) : (h, m, d, a) => { call(h, m, a); };
}
`;

function patchOrtGlue(src, dst) {
  const text = readFileSync(src, "utf8");
  const re = /\(new Function\(Object\.keys\((\w+)\),(\w+)\)\)\(\.\.\.Object\.values\(\1\)\)/g;
  const hits = [...text.matchAll(re)];
  if (hits.length !== 1 || !text.includes("return function (handle, methodName, destructorsRef, args)")) {
    throw new Error(`ORT glue changed (${hits.length} new Function sites); update the invoker patch in build_ext.mjs`);
  }
  if (/new Function|\beval\(/.test(text.replace(re, ""))) throw new Error("ORT glue has another dynamic-code site");
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, INVOKER + text.replace(re, (_, env, body) => `__panelProbeInvoker(${body},${env})`));
}

// opencv.js: craftInvokerFunction (every bound C++ function/method) and __emval_get_method_caller.
const CRAFT_INVOKER = `function craftInvokerFunction(humanName,argTypes,classType,cppInvokerFunc,cppTargetFunc,isAsync){
  var argCount=argTypes.length;
  if(argCount<2){throwBindingError("argTypes array size mismatch! Must at least get return value and 'this' types!")}
  var isClassMethodFunc=argTypes[1]!==null&&classType!==null;
  var needsDestructorStack=usesDestructorStack(argTypes);
  var returns=argTypes[0].name!=="void";
  var n=argCount-2;
  var invokerFn=function(...args){
    if(args.length!==n){throwBindingError("function "+humanName+" called with "+args.length+" arguments, expected "+n)}
    var destructors=needsDestructorStack?[]:null;
    var wired=[cppTargetFunc];
    var thisWired;
    if(isClassMethodFunc){thisWired=argTypes[1]["toWireType"](destructors,this);wired.push(thisWired)}
    var argsWired=new Array(n);
    for(var i=0;i<n;++i){argsWired[i]=argTypes[i+2]["toWireType"](destructors,args[i]);wired.push(argsWired[i])}
    var rv=cppInvokerFunc(...wired);
    if(needsDestructorStack){runDestructors(destructors)}
    else{for(var i=isClassMethodFunc?1:2;i<argTypes.length;++i){if(argTypes[i].destructorFunction!==null){argTypes[i].destructorFunction(i===1?thisWired:argsWired[i-2])}}}
    if(returns){return argTypes[0]["fromWireType"](rv)}
  };
  return createNamedFunction(humanName,invokerFn)
}`;
const METHOD_CALLER = `var invokerFunction=(function(){
  var offsets=[];for(var i=0,o=0;i<argCount;++i){offsets.push(o);o+=types[i]["argPackAdvance"]}
  return function(obj,func,destructorsRef,args){
    var a=new Array(argCount);
    for(var i=0;i<argCount;++i){a[i]=types[i].readValueFromPointer(args+offsets[i])}
    var rv=kind===0?func.call(obj,...a):kind===1?new func(...a):func.call(...a);
    if(!retType.isVoid){return emval_returnValue(retType,destructorsRef,rv)}
  }
})();`;

function patchOpencv(src, dst) {
  let text = readFileSync(src, "utf8");
  const craft = /function craftInvokerFunction\(humanName,argTypes,classType,cppInvokerFunc,cppTargetFunc,isAsync\)\{[\s\S]*?return createNamedFunction\(humanName,invokerFn\)\}/g;
  const caller = "var invokerFunction=newFunc(Function,params)(...args);";
  if ([...text.matchAll(craft)].length !== 1 || text.split(caller).length !== 2) {
    throw new Error("opencv.js embind changed; update the invoker patches in build_ext.mjs");
  }
  text = text.replace(craft, () => CRAFT_INVOKER).replace(caller, () => METHOD_CALLER);
  if (/newFunc\(Function|new Function|\beval\(/.test(text)) throw new Error("opencv.js has another dynamic-code site");
  mkdirSync(dirname(dst), { recursive: true });
  writeFileSync(dst, text);
}

const RELEASE = "https://github.com/elClie/Real-Time-Manga-Translator/releases/latest";
const missing = MODELS.filter((f) => !existsSync(join(ROOT, "models", f)));
if (missing.length) throw new Error(`models/ is missing ${missing.join(", ")}: unzip models.zip from ${RELEASE} into models/`);

rmSync(OUT, { recursive: true, force: true });
cpSync(join(ROOT, "extension"), OUT, { recursive: true });
mkdirSync(join(OUT, "pipeline"), { recursive: true });
for (const f of readdirSync(join(ROOT, "pipeline")).filter((f) => f.endsWith(".js"))) {
  copyFileSync(join(ROOT, "pipeline", f), join(OUT, "pipeline", f));
}
for (const f of VENDOR) link(join(ROOT, "node_modules", f), join(OUT, "vendor", f));
patchOrtGlue(join(ROOT, "node_modules", ORT_GLUE), join(OUT, "vendor", ORT_GLUE));
patchOpencv(join(ROOT, "node_modules", OPENCV), join(OUT, "vendor", OPENCV));
for (const f of MODELS) link(join(ROOT, "models", f), join(OUT, "models", f));

const size = MODELS.reduce((n, f) => n + statSync(join(ROOT, "models", f)).size, 0);
console.log(`built ${OUT} (models ${(size / 2 ** 20).toFixed(0)} MB, hard-linked)`);
