use std::collections::HashMap;
use std::collections::HashSet;
use std::cell::RefCell;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labels(pub Vec<LabelSymbol>);

impl Labels {
    pub fn new(mut syms: Vec<LabelSymbol>) -> Self {
        syms.sort_unstable();
        Self(syms)
    }

    pub fn join_labels(self, other: Self) -> Option<Labels> {
        let mut set: HashSet<LabelSymbol> = self.0.iter().cloned().collect();
        set.extend(other.0.iter().cloned());
        let mut joined: Vec<_> = set.into_iter().collect();
        joined.sort_unstable();
        Some(Labels(joined))
    }

    pub fn flows_to(&self, other: Self) -> bool {
        self.0.iter().all(|s| other.0.contains(s))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LabelSymbol(usize);

pub struct LabelInterner {
    string_to_sym: HashMap<String, LabelSymbol>,
    sym_to_string: Vec<String>,
}

impl LabelInterner {
    pub fn new() -> Self {
        Self {
            string_to_sym: HashMap::new(),
            sym_to_string: Vec::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> LabelSymbol {
        if let Some(ls) = self.string_to_sym.get(s) {
            *ls
        } else {
            let next_id = self.sym_to_string.len();
            self.sym_to_string.push(s.to_string());
            let ls = LabelSymbol(next_id);
            self.string_to_sym.insert(s.to_string(), ls);
            ls
        }
    }

    pub fn resolve(&self, sym: LabelSymbol) -> Option<String> {
        self.sym_to_string.get(sym.0).map(|s| s.to_string())
    }
}

thread_local! {
    static LABEL_INTERNER: RefCell<LabelInterner> = RefCell::new(LabelInterner::new());
}

pub fn intern_label(s: &str) -> LabelSymbol {
    LABEL_INTERNER.with(|interner| {
        interner.borrow_mut().intern(s)
    })
}

pub fn resolve_label(sym: LabelSymbol) -> Option<String> {
    LABEL_INTERNER.with(|interner| {
        interner.borrow().resolve(sym).map(|s| s.to_string())
    })
}
