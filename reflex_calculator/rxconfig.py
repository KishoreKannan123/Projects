import reflex as rx

config = rx.Config(
    app_name="Project",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)