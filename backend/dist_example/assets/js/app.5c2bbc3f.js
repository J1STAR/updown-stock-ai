(function (t) {
    function e(e) {
        for (var n, o, i = e[0], u = e[1], l = e[2], c = 0, p = []; c < i.length; c++) o = i[c], Object.prototype.hasOwnProperty.call(r, o) && r[o] && p.push(r[o][0]), r[o] = 0;
        for (n in u) Object.prototype.hasOwnProperty.call(u, n) && (t[n] = u[n]);
        f && f(e);
        while (p.length) p.shift()();
        return s.push.apply(s, l || []), a()
    }

    function a() {
        for (var t, e = 0; e < s.length; e++) {
            for (var a = s[e], n = !0, i = 1; i < a.length; i++) {
                var u = a[i];
                0 !== r[u] && (n = !1)
            }
            n && (s.splice(e--, 1), t = o(o.s = a[0]))
        }
        return t
    }

    var n = {}, r = {app: 0}, s = [];

    function o(e) {
        if (n[e]) return n[e].exports;
        var a = n[e] = {i: e, l: !1, exports: {}};
        return t[e].call(a.exports, a, a.exports, o), a.l = !0, a.exports
    }

    o.m = t, o.c = n, o.d = function (t, e, a) {
        o.o(t, e) || Object.defineProperty(t, e, {enumerable: !0, get: a})
    }, o.r = function (t) {
        "undefined" !== typeof Symbol && Symbol.toStringTag && Object.defineProperty(t, Symbol.toStringTag, {value: "Module"}), Object.defineProperty(t, "__esModule", {value: !0})
    }, o.t = function (t, e) {
        if (1 & e && (t = o(t)), 8 & e) return t;
        if (4 & e && "object" === typeof t && t && t.__esModule) return t;
        var a = Object.create(null);
        if (o.r(a), Object.defineProperty(a, "default", {
            enumerable: !0,
            value: t
        }), 2 & e && "string" != typeof t) for (var n in t) o.d(a, n, function (e) {
            return t[e]
        }.bind(null, n));
        return a
    }, o.n = function (t) {
        var e = t && t.__esModule ? function () {
            return t["default"]
        } : function () {
            return t
        };
        return o.d(e, "a", e), e
    }, o.o = function (t, e) {
        return Object.prototype.hasOwnProperty.call(t, e)
    }, o.p = "/";
    var i = window["webpackJsonp"] = window["webpackJsonp"] || [], u = i.push.bind(i);
    i.push = e, i = i.slice();
    for (var l = 0; l < i.length; l++) e(i[l]);
    var f = u;
    s.push([0, "chunk-vendors"]), a()
})({
    0: function (t, e, a) {
        t.exports = a("56d7")
    }, "56d7": function (t, e, a) {
        "use strict";
        a.r(e);
        a("cadf"), a("551c"), a("f751"), a("097d");
        var n = a("2b0e"), r = function () {
                var t = this, e = t.$createElement, a = t._self._c || e;
                return a("v-app", [a("v-app-bar", {attrs: {app: ""}}, [a("v-toolbar-title", {staticClass: "headline text-uppercase"}, [a("span", [t._v("Vuetify")]), a("span", {staticClass: "font-weight-light"}, [t._v("MATERIAL DESIGN")])]), a("v-spacer"), a("v-btn", {
                    attrs: {
                        text: "",
                        href: "https://github.com/vuetifyjs/vuetify/releases/latest",
                        target: "_blank"
                    }
                }, [a("span", {staticClass: "mr-2"}, [t._v("Latest Release")])])], 1), a("v-content", [a("HelloWorld")], 1)], 1)
            }, s = [], o = function () {
                var t = this, e = t.$createElement, n = t._self._c || e;
                return n("v-container", [n("v-layout", {
                    attrs: {
                        "text-center": "",
                        wrap: ""
                    }
                }, [n("v-flex", {attrs: {xs12: ""}}, [n("v-img", {
                    staticClass: "my-3",
                    attrs: {src: a("9b19"), contain: "", height: "200"}
                })], 1), n("v-flex", {attrs: {"mb-4": ""}}, [n("h1", {staticClass: "display-2 font-weight-bold mb-3"}, [t._v("\n        Welcome to Vuetify\n      ")]), n("p", {staticClass: "subheading font-weight-regular"}, [t._v("\n        For help and collaboration with other Vuetify developers,\n        "), n("br"), t._v("please join our online\n        "), n("a", {
                    attrs: {
                        href: "https://community.vuetifyjs.com",
                        target: "_blank"
                    }
                }, [t._v("Discord Community")])])]), n("v-flex", {
                    attrs: {
                        "mb-5": "",
                        xs12: ""
                    }
                }, [n("h2", {staticClass: "headline font-weight-bold mb-3"}, [t._v("What's next?")]), n("v-layout", {attrs: {"justify-center": ""}}, t._l(t.whatsNext, (function (e, a) {
                    return n("a", {
                        key: a,
                        staticClass: "subheading mx-3",
                        attrs: {href: e.href, target: "_blank"}
                    }, [t._v("\n          " + t._s(e.text) + "\n        ")])
                })), 0)], 1), n("v-flex", {
                    attrs: {
                        xs12: "",
                        "mb-5": ""
                    }
                }, [n("h2", {staticClass: "headline font-weight-bold mb-3"}, [t._v("Important Links")]), n("v-layout", {attrs: {"justify-center": ""}}, t._l(t.importantLinks, (function (e, a) {
                    return n("a", {
                        key: a,
                        staticClass: "subheading mx-3",
                        attrs: {href: e.href, target: "_blank"}
                    }, [t._v("\n          " + t._s(e.text) + "\n        ")])
                })), 0)], 1), n("v-flex", {
                    attrs: {
                        xs12: "",
                        "mb-5": ""
                    }
                }, [n("h2", {staticClass: "headline font-weight-bold mb-3"}, [t._v("Ecosystem")]), n("v-layout", {attrs: {"justify-center": ""}}, t._l(t.ecosystem, (function (e, a) {
                    return n("a", {
                        key: a,
                        staticClass: "subheading mx-3",
                        attrs: {href: e.href, target: "_blank"}
                    }, [t._v("\n          " + t._s(e.text) + "\n        ")])
                })), 0)], 1)], 1)], 1)
            }, i = [], u = {
                data: function () {
                    return {
                        ecosystem: [{
                            text: "vuetify-loader",
                            href: "https://github.com/vuetifyjs/vuetify-loader"
                        }, {text: "github", href: "https://github.com/vuetifyjs/vuetify"}, {
                            text: "awesome-vuetify",
                            href: "https://github.com/vuetifyjs/awesome-vuetify"
                        }],
                        importantLinks: [{text: "Documentation", href: "https://vuetifyjs.com"}, {
                            text: "Chat",
                            href: "https://community.vuetifyjs.com"
                        }, {text: "Made with Vuetify", href: "https://madewithvuejs.com/vuetify"}, {
                            text: "Twitter",
                            href: "https://twitter.com/vuetifyjs"
                        }, {text: "Articles", href: "https://medium.com/vuetify"}],
                        whatsNext: [{
                            text: "Explore components",
                            href: "https://vuetifyjs.com/components/api-explorer"
                        }, {
                            text: "Select a layout",
                            href: "https://vuetifyjs.com/layout/pre-defined"
                        }, {
                            text: "Frequently Asked Questions",
                            href: "https://vuetifyjs.com/getting-started/frequently-asked-questions"
                        }]
                    }
                }
            }, l = u, f = a("2877"), c = a("6544"), p = a.n(c), h = a("a523"), v = a("0e8f"), y = a("adda"), d = a("a722"),
            m = Object(f["a"])(l, o, i, !1, null, null, null), b = m.exports;
        p()(m, {VContainer: h["a"], VFlex: v["a"], VImg: y["a"], VLayout: d["a"]});
        var x = {
                name: "App", components: {HelloWorld: b}, data: function () {
                    return {}
                }
            }, g = x, _ = a("7496"), w = a("40dc"), j = a("8336"), k = a("a75b"), C = a("2fa4"), V = a("2a7f"),
            O = Object(f["a"])(g, r, s, !1, null, null, null), S = O.exports;
        p()(O, {VApp: _["a"], VAppBar: w["a"], VBtn: j["a"], VContent: k["a"], VSpacer: C["a"], VToolbarTitle: V["a"]});
        var A = a("f309");
        n["a"].use(A["a"]);
        var P = new A["a"]({icons: {iconfont: "mdi"}}), T = a("2f62"), E = a("8c4f"), L = a("bc3a"), M = a.n(L);
        n["a"].config.productionTip = !1, n["a"].use(T["a"]), n["a"].use(E["a"]), n["a"].prototype.$http = M.a, new n["a"]({
            vuetify: P,
            render: function (t) {
                return t(S)
            }
        }).$mount("#app")
    }, "9b19": function (t, e, a) {
        t.exports = a.p + "assets/img/logo.63a7d78d.svg"
    }
});
//# sourceMappingURL=app.5c2bbc3f.js.map