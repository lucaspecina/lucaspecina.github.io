---
layout: default
title: "Inicio"
---

# Lucas Pecina

AI @ Y-TEC. Research

---

## Últimos posts
<ul class="post-list">
{%- for post in site.posts limit:5 -%}
  <li>
    <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <span class="meta">{{ post.date | date: "%b %d, %Y" }}</span>
  </li>
{%- endfor -%}
</ul>